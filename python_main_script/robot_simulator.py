"""
robot_simulator.py
------------------
2-D robot simulator with real-time matplotlib visualisation.

Simulates the differential-drive robot on a map (OccupancyGrid) and
draws the robot position, path history, planned waypoints and obstacles.

No ROS required.  Dependencies: numpy, matplotlib.

Usage
-----
    from map_loader       import create_demo_map
    from diff_drive       import DiffDriveController, RobotState
    from navigator        import PurePursuitNavigator
    from zigzag_planner   import ZigzagPlanner
    from robot_simulator  import RobotSimulator

    grid    = create_demo_map()
    robot   = DiffDriveController(initial_state=RobotState(1.0, 1.0, 0.0))
    planner = ZigzagPlanner(grid)
    wps     = planner.generate()

    nav = PurePursuitNavigator(robot)
    nav.set_path(wps)

    sim = RobotSimulator(grid, robot, nav)
    sim.run()   # opens matplotlib window
"""

from __future__ import annotations
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # try TkAgg; fall back below if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow

from map_loader   import OccupancyGrid
from diff_drive   import DiffDriveController, RobotState
from navigator    import PurePursuitNavigator
from robot_config import ROBOT_RADIUS, CHASSIS_LENGTH, CHASSIS_WIDTH


class RobotSimulator:
    """
    Interactive 2-D simulator.

    Parameters
    ----------
    grid       : OccupancyGrid
    robot      : DiffDriveController
    navigator  : PurePursuitNavigator  (optional; if None you drive manually)
    dt         : float   simulation time step  (seconds)
    speed      : float   real-time factor  (1.0 = real time, >1 = faster)
    """

    def __init__(
        self,
        grid:      OccupancyGrid,
        robot:     DiffDriveController,
        navigator: Optional[PurePursuitNavigator] = None,
        dt:        float = 0.05,
        speed:     float = 5.0,
    ):
        self.grid  = grid
        self.robot = robot
        self.nav   = navigator
        self.dt    = dt
        self.speed = speed

        self._trail: List[Tuple[float, float]] = []
        self._covered: List[Tuple[float, float]] = []

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the matplotlib window and run the simulation."""
        try:
            self._setup_figure()
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            plt.close('all')
            print("\n[simulator] Stopped.")

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_figure(self) -> None:
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(10, 8))
        self._fig.canvas.manager.set_window_title("Cleaner Robot Simulator")

        xmin, xmax, ymin, ymax = self.grid.extent()
        self._ax.set_xlim(xmin, xmax)
        self._ax.set_ylim(ymin, ymax)
        self._ax.set_aspect('equal')
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_title("Cleaner Robot 2-D Simulation")

        # Map background: free=white, occupied=black, unknown=grey
        display = np.full(self.grid.prob.shape, 0.5)     # unknown = grey
        display[self.grid.free_mask] = 1.0               # free = white
        display[self.grid.occ_mask]  = 0.0               # occupied = black
        self._ax.imshow(
            display,
            cmap='gray', vmin=0, vmax=1,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            zorder=0,
        )

        # Waypoints scatter
        if self.nav and self.nav._path:
            xs = [p[0] for p in self.nav._path]
            ys = [p[1] for p in self.nav._path]
            self._wp_scatter = self._ax.scatter(xs, ys, s=8, c='cyan',
                                                zorder=2, label='Waypoints')
        else:
            self._wp_scatter = None

        # Trail line
        self._trail_line, = self._ax.plot([], [], 'b-', linewidth=1.5,
                                          zorder=3, label='Path taken')

        # Robot body patch (rectangle)
        self._robot_patch = mpatches.FancyBboxPatch(
            (0, 0), CHASSIS_LENGTH, CHASSIS_WIDTH,
            boxstyle="round,pad=0.01",
            linewidth=1.5, edgecolor='red', facecolor='orange',
            zorder=5,
        )
        self._ax.add_patch(self._robot_patch)

        # Heading arrow
        self._heading_arrow: Optional[FancyArrow] = None

        self._ax.legend(loc='upper right', fontsize=8)
        self._info_text = self._ax.text(
            0.01, 0.01, '', transform=self._ax.transAxes,
            fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6),
            zorder=10,
        )
        plt.tight_layout()
        plt.pause(0.01)

    def _loop(self) -> None:
        step = 0
        while plt.fignum_exists(self._fig.number):
            t0 = time.monotonic()

            # Advance navigation
            if self.nav and not self.nav.goal_reached():
                self.nav.compute_velocity()

            self.robot.update(self.dt)
            state = self.robot.state
            self._trail.append((state.x, state.y))

            # Collision check
            if not self.grid.is_free_world(state.x, state.y):
                print("[simulator] ⚠  Collision detected!")
                self.robot.set_velocity(0.0, 0.0)

            # Update visuals every 4 steps
            if step % 4 == 0:
                self._update_display(state)

            step += 1

            # Timing
            elapsed = time.monotonic() - t0
            sleep_t = self.dt / self.speed - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        print("[simulator] Window closed.")

    def _update_display(self, state: RobotState) -> None:
        # Trail
        if len(self._trail) >= 2:
            xs = [p[0] for p in self._trail]
            ys = [p[1] for p in self._trail]
            self._trail_line.set_data(xs, ys)

        # Robot rectangle centred on (x, y), rotated by theta
        half_l = CHASSIS_LENGTH / 2
        half_w = CHASSIS_WIDTH  / 2
        # Corner offset before rotation (front-left)
        bx = state.x - half_l * math.cos(state.theta) + half_w * math.sin(state.theta)
        by = state.y - half_l * math.sin(state.theta) - half_w * math.cos(state.theta)
        self._robot_patch.set_x(bx)
        self._robot_patch.set_y(by)
        # matplotlib patches don't support easy rotation; use transform instead
        t = (
            matplotlib.transforms.Affine2D()
            .rotate_around(state.x, state.y, state.theta)
            + self._ax.transData
        )
        # Reposition as a simple circle for simplicity if rotation is complex
        # (full transform approach)
        self._robot_patch.set_transform(t)

        # Heading arrow
        if self._heading_arrow:
            self._heading_arrow.remove()
        alen = ROBOT_RADIUS * 1.5
        self._heading_arrow = self._ax.annotate(
            '', xy=(state.x + alen * math.cos(state.theta),
                    state.y + alen * math.sin(state.theta)),
            xytext=(state.x, state.y),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            zorder=6,
        )

        # Info text
        done_str = ""
        if self.nav:
            pct = (self.nav.current_waypoint_index / max(1, self.nav.total_waypoints)) * 100
            done_str = f"  WP: {self.nav.current_waypoint_index}/{self.nav.total_waypoints} ({pct:.0f}%)"
        self._info_text.set_text(
            f"x={state.x:.2f}m  y={state.y:.2f}m  "
            f"θ={math.degrees(state.theta):.1f}°\n"
            f"v={self.robot.linear_velocity:.3f} m/s  "
            f"ω={self.robot.angular_velocity:.3f} rad/s"
            f"{done_str}"
        )

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
