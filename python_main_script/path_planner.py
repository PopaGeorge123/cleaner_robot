"""
path_planner.py
---------------
A* global path planner that works directly on an OccupancyGrid.

No ROS required.  Dependencies: numpy.

Usage
-----
    from map_loader   import create_demo_map
    from path_planner import AStarPlanner

    grid    = create_demo_map()
    planner = AStarPlanner(grid, inflation_radius=0.25)

    path = planner.plan(start=(1.0, 1.0), goal=(6.5, 4.5))
    # path is a list of (world_x, world_y) tuples, or None if no path found
"""

from __future__ import annotations
import math
import heapq
from typing import List, Optional, Tuple

import numpy as np

from map_loader  import OccupancyGrid
from robot_config import ROBOT_RADIUS


# ── Inflation ──────────────────────────────────────────────────────────────────

def _inflate_map(occ_mask: np.ndarray, inflation_px: int) -> np.ndarray:
    """
    Return a copy of occ_mask with obstacles grown by inflation_px pixels
    (simple binary dilation using a square kernel).
    """
    if inflation_px <= 0:
        return occ_mask.copy()
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * inflation_px + 1, 2 * inflation_px + 1), dtype=bool)
    return binary_dilation(occ_mask, structure=struct)


def _inflate_map_simple(occ_mask: np.ndarray, inflation_px: int) -> np.ndarray:
    """
    Inflation without scipy – iterate over occupied cells and mark a radius.
    Slower but zero extra dependencies.
    """
    if inflation_px <= 0:
        return occ_mask.copy()
    inflated = occ_mask.copy()
    rows, cols = np.where(occ_mask)
    H, W = occ_mask.shape
    for r, c in zip(rows, cols):
        r0 = max(0, r - inflation_px)
        r1 = min(H, r + inflation_px + 1)
        c0 = max(0, c - inflation_px)
        c1 = min(W, c + inflation_px + 1)
        inflated[r0:r1, c0:c1] = True
    return inflated


# ── A* planner ─────────────────────────────────────────────────────────────────

class AStarPlanner:
    """
    Grid-based A* path planner.

    Parameters
    ----------
    grid             : OccupancyGrid
    inflation_radius : float  metres  – how much to grow obstacles (robot safety margin)
    allow_diagonal   : bool   – allow diagonal moves (8-connected grid)
    """

    def __init__(
        self,
        grid:             OccupancyGrid,
        inflation_radius: float = ROBOT_RADIUS,
        allow_diagonal:   bool  = True,
    ):
        self.grid   = grid
        self.allow_diagonal = allow_diagonal

        # Build inflated obstacle mask
        inflation_px = int(math.ceil(inflation_radius / grid.resolution))
        try:
            from scipy.ndimage import binary_dilation
            struct = np.ones((2 * inflation_px + 1, 2 * inflation_px + 1), dtype=bool)
            self._blocked = binary_dilation(grid.occ_mask, structure=struct)
        except ImportError:
            self._blocked = _inflate_map_simple(grid.occ_mask, inflation_px)

        # Also block unknown cells (not free and not occupied → unknown)
        unknown = ~grid.free_mask & ~grid.occ_mask
        self._blocked |= unknown

    # ── Public ────────────────────────────────────────────────────────────────

    def plan(
        self,
        start: Tuple[float, float],
        goal:  Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal in world coordinates (metres).

        Returns
        -------
        List of (world_x, world_y) waypoints including start and goal,
        or None if no path is found.
        """
        grid = self.grid
        sr, sc = grid.world_to_pixel(*start)
        gr, gc = grid.world_to_pixel(*goal)

        if self._blocked[sr, sc]:
            print("[AStarPlanner] Warning: start cell is blocked – nudging to nearest free cell.")
            sr, sc = self._nearest_free(sr, sc)
        if self._blocked[gr, gc]:
            print("[AStarPlanner] Warning: goal cell is blocked – nudging to nearest free cell.")
            gr, gc = self._nearest_free(gr, gc)

        pixel_path = self._astar((sr, sc), (gr, gc))
        if pixel_path is None:
            return None

        world_path = [grid.pixel_to_world(r, c) for r, c in pixel_path]
        world_path = _smooth_path(world_path)
        return world_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _astar(
        self,
        start: Tuple[int, int],
        goal:  Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        H, W = self._blocked.shape

        # Priority queue: (f, g, (r, c))
        open_heap: list = []
        heapq.heappush(open_heap, (0.0, 0.0, start))

        came_from: dict = {}
        g_score:   dict = {start: 0.0}

        if self.allow_diagonal:
            neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            costs      = [math.sqrt(2), 1, math.sqrt(2), 1, 1, math.sqrt(2), 1, math.sqrt(2)]
        else:
            neighbors = [(-1,0),(0,-1),(0,1),(1,0)]
            costs      = [1, 1, 1, 1]

        while open_heap:
            _f, g, current = heapq.heappop(open_heap)

            if current == goal:
                return _reconstruct(came_from, current)

            for (dr, dc), cost in zip(neighbors, costs):
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if self._blocked[nr, nc]:
                    continue
                tentative_g = g + cost
                neighbor = (nr, nc)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    h = _heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, neighbor))

        return None  # no path

    def _nearest_free(self, r: int, c: int) -> Tuple[int, int]:
        H, W = self._blocked.shape
        for radius in range(1, 50):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not self._blocked[nr, nc]:
                        return nr, nc
        return r, c


# ── Helpers ────────────────────────────────────────────────────────────────────

def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Octile distance heuristic (admissible for 8-connected grid)."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


def _reconstruct(came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _smooth_path(
    path: List[Tuple[float, float]],
    iterations: int = 50,
    alpha: float = 0.5,
    beta: float  = 0.25,
) -> List[Tuple[float, float]]:
    """
    Gradient-descent path smoothing (keeps endpoints fixed).
    Reduces jagged grid artefacts.
    """
    if len(path) < 3:
        return path
    smoothed = [list(p) for p in path]
    n = len(smoothed)
    for _ in range(iterations):
        for i in range(1, n - 1):
            for d in range(2):
                smoothed[i][d] += (
                    alpha * (path[i][d] - smoothed[i][d])
                    + beta  * (smoothed[i-1][d] + smoothed[i+1][d] - 2 * smoothed[i][d])
                )
    return [(p[0], p[1]) for p in smoothed]
