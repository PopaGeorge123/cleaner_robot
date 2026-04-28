# Cleaner Robot – Pure Python (no ROS)

This folder contains a **complete, standalone reimplementation** of the
`articubot_one` ROS project in plain Python 3. You do **not** need ROS, Gazebo,
or any ROS toolchain to run it.

---

## File overview

| File | What it does |
|------|-------------|
| `main.py` | **Executable entry point** – interactive menu |
| `robot_config.py` | All physical constants from the URDF and YAML configs |
| `map_loader.py` | Load a ROS-style map (YAML + PGM/PNG) into NumPy arrays |
| `diff_drive.py` | Differential-drive kinematics, odometry, velocity clamping |
| `path_planner.py` | A\* global path planner on the occupancy grid |
| `zigzag_planner.py` | Lawnmower / boustrophedon coverage-path generator |
| `navigator.py` | Pure-pursuit local follower + spin/backup recovery behaviours |
| `ball_tracker.py` | HSV ball detection + follow-ball controller (OpenCV) |
| `joystick_controller.py` | Keyboard (WASD) and gamepad teleoperation |
| `robot_simulator.py` | 2-D real-time simulator with matplotlib visualisation |
| `lidar_ld06.py` | **LD06 lidar** serial driver + simulated scanner |
| `requirements.txt` | All Python dependencies |

---

## Quick start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

> On a Raspberry Pi you can omit `pygame` if you have no gamepad, and omit
> `opencv-python` if you only want path planning.
> `pyserial` is required only for the live LD06 lidar mode.

### 2 – Run

```bash
cd python_main_script
python3 main.py
```

You will see an interactive menu:

```
  1  Zigzag / coverage path planner
  2  Simulate full coverage run  (matplotlib window)
  3  A* path planning demo       (matplotlib window)
  4  Ball tracker  (webcam)
  5  Keyboard teleoperation
  6  Gamepad teleoperation
  7  Show robot configuration
  0  Quit
```

---

## Modes explained

### 1 – Zigzag coverage planner
Generates a lawnmower path over a map and saves it as a CSV file.

- Choose the built-in **demo map** (no file required) or supply a ROS map YAML.
- Set the **swath width** (brush/cleaning width in metres, default 0.5 m).
- Outputs `waypoints.csv` with columns `x,y` in world metres.

### 2 – Simulate coverage run
Opens a **matplotlib window** showing the robot driving the zigzag path in
real time.  The map, robot body (orange rectangle) and path trail (blue line)
are drawn as the simulation runs.

### 3 – A\* path planning demo
Plan a single point-to-point path on the map using the A\* algorithm with
obstacle inflation. Displays the result in a matplotlib window.

### 4 – Ball tracker (webcam)
Opens the default camera (index 0) and detects a coloured ball using HSV
thresholding (parameters from `ball_tracker_params_sim.yaml`).
The computed velocity command `(v, w)` is printed on screen.

Press **`t`** to toggle the HSV mask view for tuning.  Press **`q`** to quit.

### 5 – Keyboard teleoperation
Drive the simulated robot with **WASD** keys:

| Key | Action |
|-----|--------|
| W / ↑ | Forward |
| S / ↓ | Backward |
| A / ← | Turn left |
| D / → | Turn right |
| Space | Full stop |
| T | Toggle turbo |

### 6 – Gamepad teleoperation
Drive with a USB gamepad (mirrors `joystick.yaml`):
- Left stick Y → linear speed
- Left stick X → angular speed
- Button 6 → deadman (must hold to move)
- Button 7 → turbo

### 7 – Show robot configuration
Prints every physical constant (wheel radius, chassis size, speed limits, LD06
lidar parameters, etc.) extracted from the original URDF and YAML files.

---

## LD06 Lidar

The project uses the **LDRobot LD06** 360° laser scanner instead of the
RPLidar that is referenced in the original ROS launch file (`rplidar.launch.py`).

### LD06 specs

| Property | Value |
|----------|-------|
| Interface | UART / USB-Serial |
| Baud rate | **230 400** |
| Scan frequency | 10 Hz |
| Points per scan | 360 |
| Angular resolution | 1 ° |
| Range | 0.02 m – 12 m |
| FOV | 360 ° |
| Mount offset (from base\_link) | x = +0.122 m, z = +0.212 m |

### Connect the LD06

1. Plug the LD06 into a USB port.
2. Find the serial device:
   ```bash
   ls /dev/ttyUSB*      # Linux / Raspberry Pi
   ls /dev/tty.usbserial*  # macOS
   ```
3. Edit `robot_config.py` → `LIDAR_SERIAL_PORT` to match your device,
   **or** enter the port path when prompted by the menu.

### Menu options 8 & 9

| Option | Description |
|--------|-------------|
| **8** – Live scan viewer | Streams real data from the LD06, prints nearest obstacle distance/angle in real time |
| **9** – Simulated scan demo | Generates a synthetic 360° scan with fake obstacles and shows a polar plot (no hardware needed) |

### Use LD06 data in your own code

```python
from lidar_ld06 import LD06Driver, nearest_obstacle, polar_to_cartesian

lidar = LD06Driver(port="/dev/ttyUSB0")
lidar.start()

scan = lidar.get_scan()          # blocks until one full 360° scan arrives
print(nearest_obstacle(scan))    # (angle_deg, dist_m, intensity)

pts = polar_to_cartesian(scan)   # list of (x, y) in robot frame
lidar.stop()
```

---

## Using a real ROS map

If you have run `map_saver` in the original ROS project, copy the `.yaml` and
`.pgm` (or `.png`) files to any folder and choose option **b** when prompted
for a map path.

---

## Robot physical parameters (summary)

| Parameter | Value |
|-----------|-------|
| Wheel radius | 33 mm |
| Wheel separation | 297 mm |
| Chassis | 335 × 265 × 138 mm |
| Max linear speed | 0.26 m/s |
| Max angular speed | 1.0 rad/s |
| Robot radius (collision) | 0.22 m |
