"""
robot_config.py
---------------
All physical and tuning constants for the cleaner robot.
Values extracted directly from the URDF xacro files and YAML configs.

No ROS required.
"""

# ── Physical dimensions (from robot_core.xacro) ───────────────────────────────
CHASSIS_LENGTH  = 0.335   # m
CHASSIS_WIDTH   = 0.265   # m
CHASSIS_HEIGHT  = 0.138   # m
CHASSIS_MASS    = 1.0     # kg

WHEEL_RADIUS    = 0.033   # m
WHEEL_THICKNESS = 0.026   # m
WHEEL_MASS      = 0.05    # kg
WHEEL_SEPARATION = 0.297  # m  (distance between left and right wheel centres)
WHEEL_OFFSET_X  = 0.226   # m  (wheel centre ahead of base_link)
WHEEL_OFFSET_Y  = 0.1485  # m  (lateral offset)
WHEEL_OFFSET_Z  = 0.010   # m

CASTER_WHEEL_RADIUS = 0.01  # m

ROBOT_RADIUS    = 0.22    # m  (for collision / inflation, from nav2_params.yaml)

# ── LD06 Lidar (replaces the generic RPLidar in the original project) ──────────
# Hardware specs:  https://www.ldrobot.com/product/detail/10
LIDAR_MODEL          = "LD06"
LIDAR_SERIAL_PORT    = "/dev/ttyUSB0"   # change to your port, e.g. /dev/ttyUSB1
LIDAR_BAUD_RATE      = 230400
LIDAR_SCAN_FREQ_HZ   = 10              # nominal scan frequency (Hz)
LIDAR_SAMPLES        = 360             # points per full 360° scan
LIDAR_RANGE_MIN_M    = 0.02            # minimum valid range (m)
LIDAR_RANGE_MAX_M    = 12.0            # maximum valid range (m)
LIDAR_FOV_DEG        = 360.0           # full 360° FOV
LIDAR_ANGLE_RES_DEG  = 1.0             # angular resolution (degrees)
# Physical mount offset from base_link (same joint as lidar.xacro)
LIDAR_MOUNT_X        = 0.122           # m  forward
LIDAR_MOUNT_Y        = 0.0             # m  lateral
LIDAR_MOUNT_Z        = 0.212           # m  height
# LD06 packet protocol constants
LIDAR_PACKET_HEADER  = 0x54
LIDAR_POINTS_PER_PKT = 12             # data points per serial packet

# ── Arduino motor controller (from ros2_control.xacro) ────────────────────────
# The Arduino runs the diffdrive_arduino firmware (github.com/joshnewans/diffdrive_arduino)
# and is connected to the Raspberry Pi via USB.
ARDUINO_SERIAL_PORT    = "/dev/ttyUSB0"   # change if lidar is also on USB
ARDUINO_BAUD_RATE      = 57600
ARDUINO_TIMEOUT_MS     = 1000
ARDUINO_LOOP_RATE_HZ   = 30              # how often the controller sends commands

# Encoder resolution
# enc_counts_per_rev = full quadrature counts for ONE wheel revolution.
# With a quadrature encoder this is:  (pulses_per_rev) × 4  × (gear_ratio)
# The value 3436 comes directly from ros2_control.xacro.
# If your motors behave incorrectly (drift, wrong speed) adjust this number.
ENC_COUNTS_PER_REV     = 3436

# Derived: how many metres does the wheel travel per encoder tick?
import math as _math
METRES_PER_TICK = (2 * _math.pi * WHEEL_RADIUS) / ENC_COUNTS_PER_REV  # ≈ 0.0000603 m
MAX_LINEAR_VEL   = 0.26   # m/s
MAX_ANGULAR_VEL  = 1.0    # rad/s
MAX_LINEAR_ACC   = 2.5    # m/s²
MAX_ANGULAR_ACC  = 3.2    # rad/s²

# ── Navigation / goal tolerances ──────────────────────────────────────────────
XY_GOAL_TOLERANCE  = 0.25  # m
YAW_GOAL_TOLERANCE = 0.25  # rad

# ── Map defaults (from nav2_params.yaml / map_saver) ──────────────────────────
MAP_RESOLUTION       = 0.05   # m/pixel  (default)
FREE_THRESH          = 0.25   # probability below which cell is "free"
OCCUPIED_THRESH      = 0.65   # probability above which cell is "occupied"

# ── Coverage / zigzag planner defaults ────────────────────────────────────────
DEFAULT_SWATH_M      = 0.50   # m  – cleaning swath width (≈ brush width)
DEFAULT_MIN_SEG_M    = 0.20   # m  – ignore free segments shorter than this

# ── Ball tracker (from ball_tracker_params_sim.yaml) ──────────────────────────
BALL_HSV_MIN = (20,  42,   0)   # H, S, V  lower bound
BALL_HSV_MAX = (37, 255, 255)   # H, S, V  upper bound
BALL_SIZE_MIN = 0               # min blob area (normalised 0–100)
BALL_SIZE_MAX = 20              # max blob area (normalised 0–100)
BALL_RADIUS_M = 0.033           # m  (physical radius of the ball)
CAMERA_HFOV   = 1.089           # rad  (horizontal field of view)

ANGULAR_CHASE_MULTIPLIER = 0.7
FORWARD_CHASE_SPEED      = 0.1  # m/s
SEARCH_ANGULAR_SPEED     = 0.5  # rad/s
MAX_BALL_SIZE_THRESH     = 0.1  # stop chasing when ball occupies > 10 % of frame
FILTER_VALUE             = 0.9  # low-pass filter weight for ball position

# ── Joystick defaults (from joystick.yaml) ────────────────────────────────────
JOY_AXIS_LINEAR  = 1      # gamepad axis index for forward/back
JOY_AXIS_ANGULAR = 0      # gamepad axis index for left/right
SCALE_LINEAR     = 0.5    # normal mode scale
SCALE_LINEAR_TURBO  = 1.0
SCALE_ANGULAR    = 0.5
SCALE_ANGULAR_TURBO = 1.0
ENABLE_BUTTON    = 6      # deadman button
ENABLE_TURBO_BUTTON = 7

# ── Keyboard teleoperation step sizes ─────────────────────────────────────────
KEY_LINEAR_STEP  = 0.05   # m/s per keypress
KEY_ANGULAR_STEP = 0.1    # rad/s per keypress
