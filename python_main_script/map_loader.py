"""
map_loader.py
-------------
Load a ROS-style occupancy-grid map (YAML + PGM/PNG image) into a plain
Python/NumPy data structure.

No ROS required.  Dependencies: numpy, pillow, pyyaml.

ROS map_server conventions used here:
  - YAML keys: image, resolution, origin, negate, occupied_thresh, free_thresh
  - Pixel 0 = black = occupied (when negate=0)
  - Pixel 255 = white = free   (when negate=0)
  - probability  = 1 - (pixel / 255)   when negate=0
  - probability  = pixel / 255         when negate=1
  - cell is FREE     if prob < free_thresh
  - cell is OCCUPIED if prob > occupied_thresh
  - otherwise UNKNOWN

Usage:
    from map_loader import load_map
    grid = load_map("my_map.yaml")

    grid.free_mask          # H x W bool array – True where robot can go
    grid.occ_mask           # H x W bool array – True where obstacles are
    grid.resolution         # metres per pixel
    grid.origin             # (x, y, theta) of the map's bottom-left corner
    grid.pixel_to_world(r, c)  -> (world_x, world_y)
    grid.world_to_pixel(wx, wy) -> (row, col)
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np
import yaml
from PIL import Image

from robot_config import FREE_THRESH, OCCUPIED_THRESH


@dataclass
class OccupancyGrid:
    """Holds a loaded map and offers coordinate-conversion helpers."""

    # Raw probability array  (H x W, float32, 0=free, 1=occupied)
    prob: np.ndarray

    resolution: float           # metres per pixel
    origin: Tuple[float, float, float]  # (x, y, theta)  world coords of pixel (H-1, 0)

    free_thresh: float = FREE_THRESH
    occ_thresh: float  = OCCUPIED_THRESH

    # ── Derived masks (computed after __post_init__) ───────────────────────────
    free_mask: np.ndarray = field(init=False)
    occ_mask:  np.ndarray = field(init=False)

    def __post_init__(self):
        self.free_mask = self.prob < self.free_thresh
        self.occ_mask  = self.prob > self.occ_thresh

    @property
    def height(self) -> int:
        return self.prob.shape[0]

    @property
    def width(self) -> int:
        return self.prob.shape[1]

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    def pixel_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert (row, col) pixel indices to (world_x, world_y) in metres."""
        ox, oy, _ = self.origin
        world_x = ox + (col + 0.5) * self.resolution
        world_y = oy + ((self.height - row - 0.5) * self.resolution)
        return world_x, world_y

    def world_to_pixel(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world (x, y) metres to nearest (row, col) pixel indices."""
        ox, oy, _ = self.origin
        col = int((world_x - ox) / self.resolution - 0.5)
        row = int(self.height - (world_y - oy) / self.resolution - 0.5)
        row = max(0, min(self.height - 1, row))
        col = max(0, min(self.width  - 1, col))
        return row, col

    def is_free_world(self, world_x: float, world_y: float) -> bool:
        """Return True if the world coordinate lies in a free cell."""
        r, c = self.world_to_pixel(world_x, world_y)
        return bool(self.free_mask[r, c])

    def extent(self):
        """Return (x_min, x_max, y_min, y_max) in world metres."""
        ox, oy, _ = self.origin
        x_min = ox
        x_max = ox + self.width  * self.resolution
        y_min = oy
        y_max = oy + self.height * self.resolution
        return x_min, x_max, y_min, y_max


# ── Public loader ──────────────────────────────────────────────────────────────

def load_map(yaml_path: str) -> OccupancyGrid:
    """
    Load a ROS map YAML file and the associated image.

    Parameters
    ----------
    yaml_path : str
        Path to the .yaml produced by map_server / map_saver.

    Returns
    -------
    OccupancyGrid
    """
    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'image' not in data:
        raise ValueError(f"Map YAML '{yaml_path}' has no 'image' key.")

    img_path   = os.path.join(os.path.dirname(yaml_path), data['image'])
    resolution = float(data.get('resolution', 0.05))
    origin     = list(data.get('origin', [0.0, 0.0, 0.0]))
    negate     = int(data.get('negate', 0))
    occ_thresh = float(data.get('occupied_thresh', OCCUPIED_THRESH))
    free_thresh= float(data.get('free_thresh',     FREE_THRESH))

    # Load image as grayscale
    im  = Image.open(img_path).convert('L')
    arr = np.array(im, dtype=np.float32)

    # Convert pixel intensity → occupancy probability
    if negate == 0:
        prob = 1.0 - arr / 255.0
    else:
        prob = arr / 255.0

    # Pad origin to length 3
    while len(origin) < 3:
        origin.append(0.0)

    grid = OccupancyGrid(
        prob       = prob,
        resolution = resolution,
        origin     = tuple(origin),
        free_thresh= free_thresh,
        occ_thresh = occ_thresh,
    )

    print(f"[map_loader] Loaded map '{img_path}'  "
          f"{grid.width}×{grid.height} px  "
          f"resolution={resolution} m/px  "
          f"free cells={grid.free_mask.sum()}")
    return grid


def create_empty_map(width_m: float = 10.0, height_m: float = 10.0,
                     resolution: float = 0.05) -> OccupancyGrid:
    """
    Create a blank (all-free) map – useful for testing without a real map file.

    Parameters
    ----------
    width_m, height_m : metres
    resolution : metres per pixel
    """
    W = int(width_m  / resolution)
    H = int(height_m / resolution)
    prob = np.zeros((H, W), dtype=np.float32)   # all free
    grid = OccupancyGrid(
        prob       = prob,
        resolution = resolution,
        origin     = (0.0, 0.0, 0.0),
    )
    print(f"[map_loader] Created empty {width_m}×{height_m} m map  "
          f"({W}×{H} px)")
    return grid


def create_demo_map(resolution: float = 0.05) -> OccupancyGrid:
    """
    Build a small demo map with walls and a few obstacles – no file needed.
    The map is 8 m × 6 m with a 0.2 m outer wall and 2 rectangular obstacles.
    """
    W = int(8.0 / resolution)
    H = int(6.0 / resolution)
    prob = np.zeros((H, W), dtype=np.float32)   # all free initially

    wall_px = max(1, int(0.2 / resolution))

    # Outer walls
    prob[:wall_px, :]  = 1.0
    prob[-wall_px:, :] = 1.0
    prob[:, :wall_px]  = 1.0
    prob[:, -wall_px:] = 1.0

    # Obstacle 1 – centre-left box
    r1s, r1e = int(H*0.3), int(H*0.5)
    c1s, c1e = int(W*0.2), int(W*0.35)
    prob[r1s:r1e, c1s:c1e] = 1.0

    # Obstacle 2 – centre-right box
    r2s, r2e = int(H*0.5), int(H*0.75)
    c2s, c2e = int(W*0.55), int(W*0.7)
    prob[r2s:r2e, c2s:c2e] = 1.0

    grid = OccupancyGrid(
        prob       = prob,
        resolution = resolution,
        origin     = (0.0, 0.0, 0.0),
    )
    print(f"[map_loader] Created demo map  {W}×{H} px  "
          f"free cells={grid.free_mask.sum()}")
    return grid
