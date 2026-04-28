"""
navigator.py
------------
Pure-pursuit local path follower and navigation controller.

Takes a list of world-coordinate waypoints and a DiffDriveController,
and drives the robot along the path by computing (linear, angular) velocity
commands at each time step.

Equivalent to the Nav2 FollowPath action + DWB local planner, but without
any ROS infrastructure.

No ROS required.

Usage
-----
    from diff_drive import DiffDriveController, RobotState
    from navigator  import PurePursuitNavigator

    robot = DiffDriveController()
    robot.state = RobotState(x=0, y=0, theta=0)

    nav = PurePursuitNavigator(robot, lookahead=0.5)
    nav.set_path([(1.0, 0.0), (2.0, 1.0), (3.0, 1.0)])

    dt = 0.05
    while not nav.goal_reached():
        cmd_v, cmd_w = nav.compute_velocity()
        robot.set_velocity(cmd_v, cmd_w)
        robot.update(dt)
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple

from diff_drive   import DiffDriveController, RobotState, angle_diff, distance
from robot_config import (
    MAX_LINEAR_VEL, MAX_ANGULAR_VEL,
    XY_GOAL_TOLERANCE, YAW_GOAL_TOLERANCE,
)


class PurePursuitNavigator:
    """
    Pure-pursuit path follower.

    The robot steers toward a 'lookahead point' on the path ahead of it.
    When close enough to the final waypoint the robot is declared done.

    Parameters
    ----------
    controller      : DiffDriveController
    lookahead       : float  metres  – how far ahead to look on the path
    linear_speed    : float  m/s     – constant forward speed during tracking
    max_angular     : float  rad/s   – angular speed cap
    goal_tolerance  : float  metres  – distance to declare goal reached
    """

    def __init__(
        self,
        controller:    DiffDriveController,
        lookahead:     float = 0.5,
        linear_speed:  float = 0.15,
        max_angular:   float = MAX_ANGULAR_VEL,
        goal_tolerance: float = XY_GOAL_TOLERANCE,
    ):
        self.ctrl          = controller
        self.lookahead     = lookahead
        self.linear_speed  = min(linear_speed, MAX_LINEAR_VEL)
        self.max_angular   = max_angular
        self.goal_tolerance= goal_tolerance

        self._path: List[Tuple[float, float]] = []
        self._wp_idx: int = 0          # current target waypoint index
        self._done: bool  = True

    # ── Path management ───────────────────────────────────────────────────────

    def set_path(self, waypoints: List[Tuple[float, float]]) -> None:
        """Load a new path.  Call this before starting navigation."""
        self._path   = list(waypoints)
        self._wp_idx = 0
        self._done   = len(waypoints) == 0

    def goal_reached(self) -> bool:
        return self._done

    @property
    def current_waypoint_index(self) -> int:
        return self._wp_idx

    @property
    def total_waypoints(self) -> int:
        return len(self._path)

    # ── Main compute ─────────────────────────────────────────────────────────

    def compute_velocity(self) -> Tuple[float, float]:
        """
        Compute the next (linear, angular) velocity command.

        Returns
        -------
        (v m/s, w rad/s)  —  also calls controller.set_velocity()
        """
        if self._done or not self._path:
            self.ctrl.set_velocity(0.0, 0.0)
            return 0.0, 0.0

        pose   = self.ctrl.state
        rx, ry = pose.x, pose.y

        # Advance waypoint index past any already-reached waypoints
        while self._wp_idx < len(self._path) - 1:
            gx, gy = self._path[self._wp_idx]
            if distance(rx, ry, gx, gy) < self.goal_tolerance:
                self._wp_idx += 1
            else:
                break

        # Check final goal
        gx, gy = self._path[-1]
        if distance(rx, ry, gx, gy) < self.goal_tolerance:
            self._done = True
            self.ctrl.set_velocity(0.0, 0.0)
            return 0.0, 0.0

        # Find lookahead point
        target = self._find_lookahead_point(rx, ry)

        # Steering angle to target
        angle_to_target = math.atan2(target[1] - ry, target[0] - rx)
        heading_error   = angle_diff(angle_to_target, pose.theta)

        # Scale angular velocity proportionally to heading error
        w = max(-self.max_angular,
                min(self.max_angular, 2.0 * heading_error))

        # Slow down when turning sharply
        v = self.linear_speed * max(0.1, 1.0 - abs(heading_error) / math.pi)

        self.ctrl.set_velocity(v, w)
        return v, w

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_lookahead_point(
        self,
        rx: float, ry: float,
    ) -> Tuple[float, float]:
        """
        Walk forward along the path from the current wp index until we find a
        point at least `lookahead` metres away, or return the last point.
        """
        for i in range(self._wp_idx, len(self._path)):
            px, py = self._path[i]
            if distance(rx, ry, px, py) >= self.lookahead:
                return px, py
        return self._path[-1]


# ── Recovery behaviours (mirror Nav2 recoveries_server) ───────────────────────

class SpinRecovery:
    """Rotate in place for `duration` seconds to escape a stuck state."""

    def __init__(self, angular_vel: float = 0.5, duration: float = 3.0):
        self.angular_vel = angular_vel
        self.duration    = duration
        self._elapsed    = 0.0
        self._active     = False

    def start(self) -> None:
        self._elapsed = 0.0
        self._active  = True

    def is_done(self) -> bool:
        return not self._active

    def compute_velocity(self, dt: float) -> Tuple[float, float]:
        if not self._active:
            return 0.0, 0.0
        self._elapsed += dt
        if self._elapsed >= self.duration:
            self._active = False
            return 0.0, 0.0
        return 0.0, self.angular_vel


class BackupRecovery:
    """Drive backwards for `distance_m` metres to escape a stuck state."""

    def __init__(self, speed: float = 0.1, distance_m: float = 0.3):
        self.speed      = speed
        self.distance_m = distance_m
        self._travelled = 0.0
        self._active    = False

    def start(self) -> None:
        self._travelled = 0.0
        self._active    = True

    def is_done(self) -> bool:
        return not self._active

    def compute_velocity(self, dt: float) -> Tuple[float, float]:
        if not self._active:
            return 0.0, 0.0
        self._travelled += self.speed * dt
        if self._travelled >= self.distance_m:
            self._active = False
            return 0.0, 0.0
        return -self.speed, 0.0
