"""
zigzag_planner.py
-----------------
Lawnmower / zig-zag coverage path generator.

Reads an OccupancyGrid and produces an ordered list of world-coordinate
waypoints that cover all free space in a boustrophedon (back-and-forth)
pattern.

No ROS required.  Dependencies: numpy.

Usage
-----
    from map_loader    import load_map, create_demo_map
    from zigzag_planner import ZigzagPlanner

    grid    = create_demo_map()
    planner = ZigzagPlanner(grid, swath_m=0.5)
    waypoints = planner.generate()   # list of (x, y) in metres
    planner.save_csv(waypoints, "waypoints.csv")
"""

from __future__ import annotations
import csv
import math
from typing import List, Tuple, Optional

import numpy as np

from map_loader   import OccupancyGrid
from robot_config import DEFAULT_SWATH_M, DEFAULT_MIN_SEG_M


Waypoint = Tuple[float, float]


class ZigzagPlanner:
    """
    Generate a lawnmower coverage path over the free cells of a map.

    Parameters
    ----------
    grid        : OccupancyGrid
    swath_m     : float  metres  – row spacing (≈ cleaning brush width)
    min_seg_m   : float  metres  – discard free segments shorter than this
    direction   : str    'horizontal' (sweep rows) or 'vertical' (sweep cols)
    """

    def __init__(
        self,
        grid:      OccupancyGrid,
        swath_m:   float = DEFAULT_SWATH_M,
        min_seg_m: float = DEFAULT_MIN_SEG_M,
        direction: str   = 'horizontal',
    ):
        self.grid      = grid
        self.swath_m   = swath_m
        self.min_seg_m = min_seg_m
        self.direction = direction

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(self) -> List[Waypoint]:
        """
        Build and return the ordered waypoint list.

        Returns
        -------
        List of (world_x, world_y) tuples.
        """
        if self.direction == 'horizontal':
            waypoints = self._sweep_rows()
        else:
            waypoints = self._sweep_cols()

        waypoints = _remove_duplicates(waypoints)
        return waypoints

    def save_csv(self, waypoints: List[Waypoint], path: str) -> None:
        """Write waypoints to a CSV file with columns x,y."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for x, y in waypoints:
                writer.writerow([f'{x:.6f}', f'{y:.6f}'])
        print(f"[ZigzagPlanner] Saved {len(waypoints)} waypoints → {path}")

    @staticmethod
    def load_csv(path: str) -> List[Waypoint]:
        """Load waypoints from a CSV file (x,y columns)."""
        waypoints: List[Waypoint] = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                waypoints.append((float(row['x']), float(row['y'])))
        print(f"[ZigzagPlanner] Loaded {len(waypoints)} waypoints from {path}")
        return waypoints

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sweep_rows(self) -> List[Waypoint]:
        grid      = self.grid
        H, W      = grid.free_mask.shape
        res       = grid.resolution
        swath_px  = max(1, int(round(self.swath_m   / res)))
        min_px    = max(1, int(round(self.min_seg_m / res)))

        waypoints: List[Waypoint] = []
        reverse = False

        for row in range(0, H, swath_px):
            bool_row  = grid.free_mask[row, :]
            intervals = _free_intervals(bool_row, min_px)
            if not intervals:
                continue

            row_wps: List[Waypoint] = []
            for s, e in intervals:
                wx_s, wy = grid.pixel_to_world(row, s)
                wx_e, _  = grid.pixel_to_world(row, e)
                row_wps.append((wx_s, wy))
                row_wps.append((wx_e, wy))

            if reverse:
                row_wps = _reverse_segment_list(row_wps)
            waypoints.extend(row_wps)
            reverse = not reverse

        return waypoints

    def _sweep_cols(self) -> List[Waypoint]:
        grid      = self.grid
        H, W      = grid.free_mask.shape
        res       = grid.resolution
        swath_px  = max(1, int(round(self.swath_m   / res)))
        min_px    = max(1, int(round(self.min_seg_m / res)))

        waypoints: List[Waypoint] = []
        reverse = False

        for col in range(0, W, swath_px):
            bool_col  = grid.free_mask[:, col]
            intervals = _free_intervals(bool_col, min_px)
            if not intervals:
                continue

            col_wps: List[Waypoint] = []
            for s, e in intervals:
                wx, wy_s = grid.pixel_to_world(s, col)
                _,  wy_e = grid.pixel_to_world(e, col)
                col_wps.append((wx, wy_s))
                col_wps.append((wx, wy_e))

            if reverse:
                col_wps = _reverse_segment_list(col_wps)
            waypoints.extend(col_wps)
            reverse = not reverse

        return waypoints


# ── Helpers ────────────────────────────────────────────────────────────────────

def _free_intervals(
    bool_line: np.ndarray,
    min_px: int,
) -> List[Tuple[int, int]]:
    """
    Find contiguous True runs in a 1-D boolean array.

    Returns list of (start_idx, end_idx) inclusive, filtered to length >= min_px.
    """
    if bool_line.sum() == 0:
        return []
    padded = np.concatenate([[False], bool_line, [False]])
    diffs  = np.diff(padded.astype(np.int8))
    starts = np.where(diffs ==  1)[0]
    ends   = np.where(diffs == -1)[0] - 1
    return [
        (int(s), int(e))
        for s, e in zip(starts, ends)
        if (e - s + 1) >= min_px
    ]


def _reverse_segment_list(wps: List[Waypoint]) -> List[Waypoint]:
    """
    Reverse the traversal direction of a list of (start, end) pairs.
    Pairs are assumed to come in pairs: [s0, e0, s1, e1, ...].
    """
    result: List[Waypoint] = []
    # Iterate pairs in reverse order, and swap start/end within each pair
    n = len(wps)
    for i in range(n - 2, -1, -2):
        result.append(wps[i + 1])   # end becomes first
        result.append(wps[i])       # start becomes second
    return result


def _remove_duplicates(
    wps: List[Waypoint],
    tol: float = 1e-4,
) -> List[Waypoint]:
    """Remove consecutive near-duplicate waypoints."""
    if not wps:
        return wps
    cleaned = [wps[0]]
    for x, y in wps[1:]:
        lx, ly = cleaned[-1]
        if abs(x - lx) > tol or abs(y - ly) > tol:
            cleaned.append((x, y))
    return cleaned
