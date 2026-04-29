#!/usr/bin/env python3
"""
main.py
-------
Entry point for the Cleaner Robot – NO ROS required.

Run this file directly:
    python3 main.py

You will be presented with an interactive menu to choose a mode:

  1. Zigzag coverage planner  – load / generate a map and produce waypoints CSV
  2. Simulate coverage run    – visualise the robot driving the zigzag path
  3. A* path planning demo    – plan a single point-to-point path on the map
  4. Ball tracker (camera)    – real-time ball detection via webcam
  5. Keyboard teleoperation   – drive the robot manually with WASD keys
  6. Gamepad teleoperation    – drive the robot with a USB gamepad
  7. Show robot config        – print all physical parameters
  0. Quit

Dependencies (install once):
    pip install numpy pillow pyyaml matplotlib scipy pynput pygame opencv-python
"""

import sys
import os

# Make sure imports from this folder work regardless of working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Lazy imports (only imported when the mode is selected) ────────────────────

def _import_check(pkg: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(pkg) is not None


# ── Menu ──────────────────────────────────────────────────────────────────────

BANNER = r"""
 ██████╗██╗     ███████╗ █████╗ ███╗   ██╗███████╗██████╗
██╔════╝██║     ██╔════╝██╔══██╗████╗  ██║██╔════╝██╔══██╗
██║     ██║     █████╗  ███████║██╔██╗ ██║█████╗  ██████╔╝
██║     ██║     ██╔══╝  ██╔══██║██║╚██╗██║██╔══╝  ██╔══██╗
╚██████╗███████╗███████╗██║  ██║██║ ╚████║███████╗██║  ██║
 ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
          R O B O T   –   P u r e   P y t h o n
"""

MENU = """
┌─────────────────────────────────────────────────────────┐
│                        MAIN MENU                        │
├─────────────────────────────────────────────────────────┤
│  1  Zigzag / coverage path planner                      │
│  2  Simulate full coverage run  (matplotlib)            │
│  3  A* path planning demo       (matplotlib)            │
│  4  Ball tracker (camera – needs opencv-python)         │
│  5  Keyboard teleoperation                              │
│  6  Gamepad teleoperation  (needs pygame)               │
│  7  Show robot configuration                            │
│  8  LD06 lidar – live scan viewer  (needs pyserial)     │
│  9  LD06 lidar – simulated scan demo                    │
│  a  Arduino – encoder calibration wizard                │
│  b  Arduino – live odometry monitor                     │
│  0  Quit                                                │
└─────────────────────────────────────────────────────────┘
"""


def prompt_map() -> object:
    """Ask user for a map YAML path or use the built-in demo map."""
    print("\n  Map options:")
    print("    a) Use built-in DEMO map (no file needed)")
    print("    b) Enter path to a ROS map YAML file")
    choice = input("  Choice [a]: ").strip().lower() or 'a'

    from map_loader import create_demo_map, load_map

    if choice == 'b':
        path = input("  Map YAML path: ").strip()
        if not os.path.isfile(path):
            print(f"  ✗ File not found: {path} – falling back to demo map.")
            return create_demo_map()
        return load_map(path)
    else:
        return create_demo_map()


# ── Mode 1: Zigzag planner ────────────────────────────────────────────────────

def mode_zigzag() -> None:
    print("\n── Zigzag Coverage Planner ──")
    grid = prompt_map()

    swath = input("  Swath width in metres [0.5]: ").strip()
    swath = float(swath) if swath else 0.5

    min_seg = input("  Min segment length metres [0.2]: ").strip()
    min_seg = float(min_seg) if min_seg else 0.2

    direction = input("  Direction  h=horizontal  v=vertical [h]: ").strip().lower()
    direction = 'vertical' if direction == 'v' else 'horizontal'

    out_csv = input("  Output CSV [waypoints.csv]: ").strip() or "waypoints.csv"

    from zigzag_planner import ZigzagPlanner
    planner = ZigzagPlanner(grid, swath_m=swath, min_seg_m=min_seg,
                            direction=direction)
    wps = planner.generate()

    if not wps:
        print("  ✗ No waypoints generated – try a smaller swath or a larger map.")
        return

    # Save relative to CWD
    planner.save_csv(wps, out_csv)
    print(f"  ✓ {len(wps)} waypoints saved to '{out_csv}'")

    # Quick stats
    import math
    total_dist = sum(
        math.hypot(wps[i+1][0]-wps[i][0], wps[i+1][1]-wps[i][1])
        for i in range(len(wps)-1)
    )
    print(f"  Total path length: {total_dist:.2f} m")


# ── Mode 2: Simulate coverage run ─────────────────────────────────────────────

def mode_simulate_coverage() -> None:
    if not _import_check('matplotlib'):
        print("  ✗ matplotlib is not installed.  Run: pip install matplotlib")
        return

    print("\n── Simulate Coverage Run ──")
    grid = prompt_map()

    swath = input("  Swath width metres [0.5]: ").strip()
    swath = float(swath) if swath else 0.5

    speed_factor = input("  Simulation speed factor [10]: ").strip()
    speed_factor = float(speed_factor) if speed_factor else 10.0

    from zigzag_planner   import ZigzagPlanner
    from diff_drive       import DiffDriveController, RobotState
    from navigator        import PurePursuitNavigator
    from robot_simulator  import RobotSimulator

    # Start position: first free cell near origin
    start_x, start_y = _find_start(grid)
    robot  = DiffDriveController(initial_state=RobotState(start_x, start_y, 0.0))

    planner = ZigzagPlanner(grid, swath_m=swath)
    wps     = planner.generate()

    if not wps:
        print("  ✗ No waypoints generated.")
        return

    print(f"  Generated {len(wps)} waypoints.  Close the window to stop.")

    nav = PurePursuitNavigator(robot, lookahead=swath * 1.5, linear_speed=0.18)
    nav.set_path(wps)

    sim = RobotSimulator(grid, robot, nav, dt=0.05, speed=speed_factor)
    sim.run()


# ── Mode 3: A* planning demo ──────────────────────────────────────────────────

def mode_astar_demo() -> None:
    if not _import_check('matplotlib'):
        print("  ✗ matplotlib is not installed.  Run: pip install matplotlib")
        return

    print("\n── A* Path Planning Demo ──")
    grid = prompt_map()

    xmin, xmax, ymin, ymax = grid.extent()
    print(f"  Map extent:  x=[{xmin:.1f}, {xmax:.1f}]  y=[{ymin:.1f}, {ymax:.1f}]")

    def prompt_point(name: str, default: tuple) -> tuple:
        raw = input(f"  {name} (x y) [{default[0]} {default[1]}]: ").strip()
        if raw:
            parts = raw.split()
            return float(parts[0]), float(parts[1])
        return default

    start = prompt_point("Start", (xmin + 0.5, ymin + 0.5))
    goal  = prompt_point("Goal",  (xmax - 0.5, ymax - 0.5))

    from path_planner import AStarPlanner
    astar = AStarPlanner(grid)
    print("  Planning …")
    path = astar.plan(start, goal)

    if path is None:
        print("  ✗ No path found.")
        return

    print(f"  ✓ Path found: {len(path)} waypoints")
    _plot_path(grid, path, start, goal, title="A* Path Planning")


def _plot_path(grid, path, start, goal, title="Path"):
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import numpy as np

    xmin, xmax, ymin, ymax = grid.extent()
    display = np.full(grid.prob.shape, 0.5)
    display[grid.free_mask] = 1.0
    display[grid.occ_mask]  = 0.0

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display, cmap='gray', vmin=0, vmax=1,
              origin='lower', extent=[xmin, xmax, ymin, ymax])

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, 'b-', linewidth=2, label='Path')
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0],  goal[1],  'r*', markersize=12, label='Goal')
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


# ── Mode 4: Ball tracker ───────────────────────────────────────────────────────

def mode_ball_tracker() -> None:
    if not _import_check('cv2'):
        print("  ✗ opencv-python is not installed.  Run: pip install opencv-python")
        return

    print("\n── Ball Tracker (webcam) ──")
    cam_id = input("  Camera index [0]: ").strip()
    cam_id = int(cam_id) if cam_id else 0

    tuning = input("  Show HSV tuning mask? [n]: ").strip().lower() == 'y'

    from ball_tracker import run_live_tracker
    run_live_tracker(camera_index=cam_id, tuning_mode=tuning)


# ── Mode 5: Keyboard teleop ───────────────────────────────────────────────────

def mode_keyboard_teleop() -> None:
    print("\n── Keyboard Teleoperation (REAL ROBOT) ──")
    if not _import_check('serial'):
        print("  ✗ pyserial not installed.  Run: pip install pyserial")
        return

    from robot_config import ARDUINO_SERIAL_PORT
    port = input(f"  Arduino serial port [{ARDUINO_SERIAL_PORT}]: ").strip() or ARDUINO_SERIAL_PORT

    from arduino_driver import ArduinoDriver
    from joystick_controller import KeyboardController

    driver = ArduinoDriver(port=port)
    driver.start()
    driver.reset_encoders()

    print("  WASD to move, SPACE to stop, Ctrl+C to quit")
    print("  (pynput required: pip install pynput)")

    try:
        from pynput import keyboard as kb
        import time

        v, w = 0.0, 0.0
        running = True

        def on_press(key):
            nonlocal v, w
            try:
                k = key.char.lower() if hasattr(key, 'char') and key.char else None
            except AttributeError:
                k = None
            if k == 'w' or key == kb.Key.up:    v = min(0.20, v + 0.05)
            elif k == 's' or key == kb.Key.down: v = max(-0.20, v - 0.05)
            elif k == 'a' or key == kb.Key.left: w = min(1.0, w + 0.1)
            elif k == 'd' or key == kb.Key.right: w = max(-1.0, w - 0.1)
            elif key == kb.Key.space: v = 0.0; w = 0.0

        def on_release(key):
            nonlocal running
            if key == kb.Key.esc:
                running = False
                return False

        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while running:
            driver.set_velocity(v, w)
            pose = driver.odometry
            print(f"\r  v={v:+.2f}m/s  w={w:+.2f}rad/s  "
                  f"x={pose.x:+.2f}  y={pose.y:+.2f}", end='', flush=True)
            time.sleep(0.05)

    except ImportError:
        print("  pynput not found, install it: pip install pynput")
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        driver.close()
        print("\n  Stopped.")
        
# ── Mode 6: Gamepad teleop ────────────────────────────────────────────────────

def mode_gamepad_teleop() -> None:
    if not _import_check('pygame'):
        print("  ✗ pygame is not installed.  Run: pip install pygame")
        return

    print("\n── Gamepad Teleoperation ──")
    from diff_drive          import DiffDriveController, RobotState
    from joystick_controller import GamepadController

    robot = DiffDriveController(initial_state=RobotState(0.0, 0.0, 0.0))
    ctrl  = GamepadController(robot)
    ctrl.run()


# ── Mode 7: Show config ───────────────────────────────────────────────────────

def mode_show_config() -> None:
    import robot_config as cfg
    print("\n── Robot Configuration ──")
    for name in sorted(dir(cfg)):
        if name.startswith('_'):
            continue
        val = getattr(cfg, name)
        if not callable(val):
            print(f"  {name:<35} = {val}")


# ── Mode 8: LD06 live scan viewer ─────────────────────────────────────────────

def mode_ld06_live() -> None:
    if not _import_check('serial'):
        print("  ✗ pyserial is not installed.  Run: pip install pyserial")
        return

    from robot_config import LIDAR_SERIAL_PORT
    port = input(f"  Serial port [{LIDAR_SERIAL_PORT}]: ").strip() or LIDAR_SERIAL_PORT

    print(f"\n── LD06 Live Scan Viewer ({port}) ──")
    print("  Press Ctrl+C to stop.\n")

    from lidar_ld06 import LD06Driver, nearest_obstacle

    lidar = LD06Driver(port=port)
    lidar.start()
    try:
        scan_count = 0
        while True:
            scan = lidar.get_scan(timeout=3.0)
            if scan is None:
                print("  [timeout] No scan received – check cable / port.")
                continue
            scan_count += 1
            nearest = nearest_obstacle(scan)
            n_str = (f"{nearest[1]:.3f} m @ {nearest[0]:.1f}°"
                     if nearest else "—")
            print(f"  Scan #{scan_count:4d}  points={len(scan):3d}  "
                  f"nearest={n_str}", end='\r', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        lidar.stop()
        print()


# ── Mode 9: LD06 simulated scan demo ─────────────────────────────────────────

def mode_ld06_sim() -> None:
    print("\n── LD06 Simulated Scan Demo ──")
    from lidar_ld06 import demo_simulated
    demo_simulated()


# ── Mode a: Arduino encoder calibration ──────────────────────────────────────

def mode_arduino_calibrate() -> None:
    if not _import_check('serial'):
        print("  ✗ pyserial is not installed.  Run: pip install pyserial")
        return
    from robot_config import ARDUINO_SERIAL_PORT
    port = input(f"  Arduino serial port [{ARDUINO_SERIAL_PORT}]: ").strip() or ARDUINO_SERIAL_PORT
    from arduino_driver import run_encoder_tuning
    run_encoder_tuning(port)


# ── Mode b: Arduino live odometry monitor ────────────────────────────────────

def mode_arduino_monitor() -> None:
    if not _import_check('serial'):
        print("  ✗ pyserial is not installed.  Run: pip install pyserial")
        return

    from robot_config import ARDUINO_SERIAL_PORT, ENC_COUNTS_PER_REV, METRES_PER_TICK
    port = input(f"  Arduino serial port [{ARDUINO_SERIAL_PORT}]: ").strip() or ARDUINO_SERIAL_PORT

    print(f"\n── Arduino Live Odometry Monitor ({port}) ──")
    print(f"  ENC_COUNTS_PER_REV = {ENC_COUNTS_PER_REV}   "
          f"METRES_PER_TICK = {METRES_PER_TICK:.8f} m")
    print("  Press Ctrl+C to stop.\n")

    from arduino_driver import ArduinoDriver
    import math

    driver = ArduinoDriver(port=port)
    driver.start()
    driver.reset_encoders()

    try:
        while True:
            l, r = driver.read_encoders()
            pose  = driver.odometry
            print(
                f"  enc L={l:7d}  R={r:7d}  |  "
                f"x={pose.x:+.3f}m  y={pose.y:+.3f}m  "
                f"θ={math.degrees(pose.theta):+6.1f}°",
                end='\r', flush=True,
            )
            import time; time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        driver.close()
        print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_start(grid) -> tuple:
    """Return the world coordinate of the first free cell near the centre."""
    import numpy as np
    rows, cols = np.where(grid.free_mask)
    if len(rows) == 0:
        return grid.pixel_to_world(grid.height // 2, grid.width // 2)
    idx = len(rows) // 4          # not the very corner
    return grid.pixel_to_world(int(rows[idx]), int(cols[idx]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(BANNER)

    handlers = {
        '1': mode_zigzag,
        '2': mode_simulate_coverage,
        '3': mode_astar_demo,
        '4': mode_ball_tracker,
        '5': mode_keyboard_teleop,
        '6': mode_gamepad_teleop,
        '7': mode_show_config,
        '8': mode_ld06_live,
        '9': mode_ld06_sim,
        'a': mode_arduino_calibrate,
        'b': mode_arduino_monitor,
        '0': None,
    }

    while True:
        print(MENU)
        choice = input("Select mode: ").strip()

        if choice == '0':
            print("Bye!")
            break

        handler = handlers.get(choice)
        if handler is None:
            print(f"  Unknown option '{choice}'")
            continue

        try:
            handler()
        except KeyboardInterrupt:
            print("\n  (interrupted)")
        except Exception as exc:
            print(f"\n  ✗ Error: {exc}")
            import traceback
            traceback.print_exc()

        input("\n  Press Enter to return to menu…")


if __name__ == '__main__':
    main()
