"""
diff_drive.py
-------------
Differential-drive robot kinematics, odometry and velocity clamping.

No ROS required.

A differential-drive robot has two wheels side-by-side.
  - left wheel speed  v_l  (m/s)
  - right wheel speed v_r  (m/s)
  - wheel separation  L    (m)
  - wheel radius      R    (m)

Forward kinematics
  linear  v = (v_r + v_l) / 2
  angular w = (v_r - v_l) / L

Inverse kinematics
  v_l = v - w * L/2
  v_r = v + w * L/2

Odometry integration (Euler step, dt seconds)
  x   += v * cos(theta) * dt
  y   += v * sin(theta) * dt
  theta += w * dt
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

from robot_config import (
    WHEEL_RADIUS, WHEEL_SEPARATION,
    MAX_LINEAR_VEL, MAX_ANGULAR_VEL,
    MAX_LINEAR_ACC, MAX_ANGULAR_ACC,
)


@dataclass
class RobotState:
    """2-D pose of the robot in world frame."""
    x:     float = 0.0   # metres
    y:     float = 0.0   # metres
    theta: float = 0.0   # radians  (positive = counter-clockwise)

    def __repr__(self):
        return (f"RobotState(x={self.x:.3f}, y={self.y:.3f}, "
                f"theta={math.degrees(self.theta):.1f}°)")


class DiffDriveController:
    """
    Manages the differential-drive robot:
    - Accepts (linear, angular) velocity commands
    - Clamps to physical limits
    - Applies smooth acceleration
    - Integrates odometry

    Parameters
    ----------
    wheel_radius     : float  metres
    wheel_separation : float  metres
    max_lin_vel      : float  m/s
    max_ang_vel      : float  rad/s
    max_lin_acc      : float  m/s²
    max_ang_acc      : float  rad/s²
    initial_state    : RobotState  (default origin)
    """

    def __init__(
        self,
        wheel_radius:     float = WHEEL_RADIUS,
        wheel_separation: float = WHEEL_SEPARATION,
        max_lin_vel:      float = MAX_LINEAR_VEL,
        max_ang_vel:      float = MAX_ANGULAR_VEL,
        max_lin_acc:      float = MAX_LINEAR_ACC,
        max_ang_acc:      float = MAX_ANGULAR_ACC,
        initial_state:    RobotState | None = None,
    ):
        self.R  = wheel_radius
        self.L  = wheel_separation
        self.max_lin = max_lin_vel
        self.max_ang = max_ang_vel
        self.max_lin_acc = max_lin_acc
        self.max_ang_acc = max_ang_acc

        self.state = initial_state or RobotState()

        # Current actual velocities (after acc limiting)
        self._v_actual: float = 0.0   # m/s
        self._w_actual: float = 0.0   # rad/s

    # ── Velocity command ───────────────────────────────────────────────────────

    def set_velocity(self, linear: float, angular: float) -> None:
        """
        Set desired (linear m/s, angular rad/s) velocity command.
        Values are clamped to physical limits.  Call update() each time-step
        to actually move the robot.
        """
        self._v_cmd = max(-self.max_lin, min(self.max_lin, linear))
        self._w_cmd = max(-self.max_ang, min(self.max_ang, angular))

    def stop(self) -> None:
        """Immediate stop command."""
        self._v_cmd = 0.0
        self._w_cmd = 0.0

    # ── Wheel speeds ───────────────────────────────────────────────────────────

    def wheel_speeds(self, linear: float, angular: float):
        """
        Convert (linear, angular) → (left_wheel_rad_s, right_wheel_rad_s).

        Returns
        -------
        (omega_left, omega_right)  in rad/s
        """
        v_l = linear - angular * self.L / 2.0
        v_r = linear + angular * self.L / 2.0
        omega_l = v_l / self.R
        omega_r = v_r / self.R
        return omega_l, omega_r

    # ── Update / integration ───────────────────────────────────────────────────

    def update(self, dt: float) -> RobotState:
        """
        Advance simulation by dt seconds.
        Applies acceleration limiting then integrates odometry.

        Parameters
        ----------
        dt : float   time step in seconds

        Returns
        -------
        Updated RobotState
        """
        # Acceleration limiting
        dv_max = self.max_lin_acc * dt
        dw_max = self.max_ang_acc * dt

        v_cmd = getattr(self, '_v_cmd', 0.0)
        w_cmd = getattr(self, '_w_cmd', 0.0)

        dv = v_cmd - self._v_actual
        dw = w_cmd - self._w_actual

        self._v_actual += max(-dv_max, min(dv_max, dv))
        self._w_actual += max(-dw_max, min(dw_max, dw))

        # Euler odometry integration
        v = self._v_actual
        w = self._w_actual
        self.state.x     += v * math.cos(self.state.theta) * dt
        self.state.y     += v * math.sin(self.state.theta) * dt
        self.state.theta += w * dt
        self.state.theta  = _wrap_angle(self.state.theta)

        return self.state

    # ── Convenience getters ───────────────────────────────────────────────────

    @property
    def pose(self) -> RobotState:
        return self.state

    @property
    def linear_velocity(self) -> float:
        return self._v_actual

    @property
    def angular_velocity(self) -> float:
        return self._w_actual


# ── Helpers ────────────────────────────────────────────────────────────────────

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle


def angle_diff(a: float, b: float) -> float:
    """Shortest signed angular difference a - b, in [-π, π]."""
    return _wrap_angle(a - b)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(x2 - x1, y2 - y1)
