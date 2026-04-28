"""
joystick_controller.py
----------------------
Keyboard and gamepad teleoperation for the cleaner robot.

Mirrors the ROS joy_node + teleop_twist_joy pipeline, but without ROS.

No ROS required.

Keyboard mode
-------------
Uses the `pynput` library (pip install pynput) for non-blocking key reads.

    WASD / arrow keys  – move
    Q / E              – turn left / right
    SPACE              – stop
    T                  – turbo toggle
    Ctrl+C             – quit

Gamepad mode
------------
Uses `pygame` (pip install pygame) to read a gamepad/joystick.
Axis mapping mirrors joystick.yaml:
    Axis 1 → linear  (left stick Y)
    Axis 0 → angular (left stick X)
    Button 6 → deadman (must hold to move)
    Button 7 → turbo

Usage
-----
    from joystick_controller import KeyboardController
    from diff_drive          import DiffDriveController

    robot = DiffDriveController()
    ctrl  = KeyboardController(robot)
    ctrl.run()          # blocks; Ctrl+C to quit

    # Or non-blocking step:
    ctrl.start()
    while True:
        v, w = ctrl.get_velocity()
        robot.set_velocity(v, w)
        time.sleep(0.05)
    ctrl.stop()
"""

from __future__ import annotations
import math
import time
import threading
from typing import Tuple

from diff_drive   import DiffDriveController
from robot_config import (
    KEY_LINEAR_STEP, KEY_ANGULAR_STEP,
    SCALE_LINEAR, SCALE_LINEAR_TURBO,
    SCALE_ANGULAR, SCALE_ANGULAR_TURBO,
    JOY_AXIS_LINEAR, JOY_AXIS_ANGULAR,
    ENABLE_BUTTON, ENABLE_TURBO_BUTTON,
    MAX_LINEAR_VEL, MAX_ANGULAR_VEL,
)


# ── Keyboard controller ───────────────────────────────────────────────────────

class KeyboardController:
    """
    Non-blocking keyboard teleoperation.

    Key bindings
    ------------
    W / ↑     : increase forward speed
    S / ↓     : increase backward speed
    A / ←     : turn left
    D / →     : turn right
    SPACE     : full stop
    T         : toggle turbo mode
    Ctrl+C    : quit
    """

    HELP = """
┌─────────────────────────────────────────┐
│         KEYBOARD TELEOPERATION          │
│                                         │
│   W / ↑   : forward                    │
│   S / ↓   : backward                   │
│   A / ←   : turn left                  │
│   D / →   : turn right                 │
│   SPACE   : stop                        │
│   T       : toggle turbo               │
│   Ctrl+C  : quit                        │
└─────────────────────────────────────────┘
"""

    def __init__(self, controller: DiffDriveController | None = None):
        self.ctrl   = controller or DiffDriveController()
        self._v     = 0.0
        self._w     = 0.0
        self._turbo = False
        self._running = False

    # ── Public ────────────────────────────────────────────────────────────────

    def get_velocity(self) -> Tuple[float, float]:
        """Return the current (linear m/s, angular rad/s) command."""
        return self._v, self._w

    def start(self) -> None:
        """Start the key-listener thread (non-blocking)."""
        self._running = True
        self._thread  = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        """Blocking loop: drive the robot via keyboard until Ctrl+C."""
        print(self.HELP)
        self.start()
        try:
            while self._running:
                v, w = self.get_velocity()
                print(f"\r  v={v:+.3f} m/s   w={w:+.3f} rad/s   "
                      f"{'TURBO' if self._turbo else '     '}   ", end='', flush=True)
                self.ctrl.set_velocity(v, w)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.ctrl.set_velocity(0.0, 0.0)
            self.stop()
            print("\n[keyboard] Stopped.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _listen_loop(self) -> None:
        try:
            from pynput import keyboard as kb

            def on_press(key):
                lin_step = KEY_LINEAR_STEP  * (SCALE_LINEAR_TURBO if self._turbo else SCALE_LINEAR)
                ang_step = KEY_ANGULAR_STEP * (SCALE_ANGULAR_TURBO if self._turbo else SCALE_ANGULAR)
                try:
                    k = key.char.lower() if hasattr(key, 'char') and key.char else None
                except AttributeError:
                    k = None

                if k == 'w' or key == kb.Key.up:
                    self._v = min(MAX_LINEAR_VEL, self._v + lin_step)
                elif k == 's' or key == kb.Key.down:
                    self._v = max(-MAX_LINEAR_VEL, self._v - lin_step)
                elif k == 'a' or key == kb.Key.left:
                    self._w = min(MAX_ANGULAR_VEL, self._w + ang_step)
                elif k == 'd' or key == kb.Key.right:
                    self._w = max(-MAX_ANGULAR_VEL, self._w - ang_step)
                elif key == kb.Key.space:
                    self._v = 0.0
                    self._w = 0.0
                elif k == 't':
                    self._turbo = not self._turbo

            def on_release(key):
                if key == kb.Key.esc:
                    self._running = False
                    return False

            with kb.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()

        except ImportError:
            print("[keyboard] pynput not installed.  Falling back to input() mode.")
            self._input_fallback()

    def _input_fallback(self) -> None:
        """Simple line-input fallback when pynput is not available."""
        print("Commands: w=forward s=back a=left d=right space=stop t=turbo q=quit")
        while self._running:
            cmd = input("> ").strip().lower()
            lin_step = KEY_LINEAR_STEP  * (SCALE_LINEAR_TURBO if self._turbo else SCALE_LINEAR)
            ang_step = KEY_ANGULAR_STEP * (SCALE_ANGULAR_TURBO if self._turbo else SCALE_ANGULAR)
            if cmd == 'w':   self._v = min(MAX_LINEAR_VEL,   self._v + lin_step)
            elif cmd == 's': self._v = max(-MAX_LINEAR_VEL,  self._v - lin_step)
            elif cmd == 'a': self._w = min(MAX_ANGULAR_VEL,  self._w + ang_step)
            elif cmd == 'd': self._w = max(-MAX_ANGULAR_VEL, self._w - ang_step)
            elif cmd == ' ' or cmd == 'stop': self._v = 0.0; self._w = 0.0
            elif cmd == 't': self._turbo = not self._turbo
            elif cmd == 'q': self._running = False; break


# ── Gamepad controller ────────────────────────────────────────────────────────

class GamepadController:
    """
    Gamepad / joystick teleoperation using pygame.

    Axis mapping from joystick.yaml:
      axis_linear  = 1   (left stick Y,  up = forward)
      axis_angular = 0   (left stick X,  left = turn left)

    Buttons:
      enable_button       = 6  (deadman – must hold)
      enable_turbo_button = 7  (hold for turbo)
    """

    def __init__(self, controller: DiffDriveController | None = None, joystick_id: int = 0):
        self.ctrl     = controller or DiffDriveController()
        self.joy_id   = joystick_id
        self._v       = 0.0
        self._w       = 0.0

    def run(self) -> None:
        """Blocking loop – Ctrl+C to quit."""
        try:
            import pygame
        except ImportError:
            print("[gamepad] pygame not installed.  Run: pip install pygame")
            return

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[gamepad] No joystick detected.")
            pygame.quit()
            return

        joy = pygame.joystick.Joystick(self.joy_id)
        joy.init()
        print(f"[gamepad] Using joystick: {joy.get_name()}")

        clock = pygame.time.Clock()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

                enabled = joy.get_button(ENABLE_BUTTON)
                turbo   = joy.get_button(ENABLE_TURBO_BUTTON)

                if enabled:
                    raw_lin = -joy.get_axis(JOY_AXIS_LINEAR)   # invert Y axis
                    raw_ang = -joy.get_axis(JOY_AXIS_ANGULAR)

                    lin_scale = SCALE_LINEAR_TURBO  if turbo else SCALE_LINEAR
                    ang_scale = SCALE_ANGULAR_TURBO if turbo else SCALE_ANGULAR

                    self._v = raw_lin * lin_scale
                    self._w = raw_ang * ang_scale
                else:
                    self._v = 0.0
                    self._w = 0.0

                self.ctrl.set_velocity(self._v, self._w)
                print(f"\r  v={self._v:+.3f}  w={self._w:+.3f}  "
                      f"{'TURBO' if turbo else '     '}", end='', flush=True)
                clock.tick(20)
        except KeyboardInterrupt:
            pass
        finally:
            self.ctrl.set_velocity(0.0, 0.0)
            pygame.quit()
            print("\n[gamepad] Stopped.")

    def get_velocity(self) -> Tuple[float, float]:
        return self._v, self._w
