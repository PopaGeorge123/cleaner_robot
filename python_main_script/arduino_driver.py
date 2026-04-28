"""
arduino_driver.py
-----------------
Serial driver for the ROSArduinoBridge firmware running on the Arduino.

Firmware source: ROSArduinoBridge/ROSArduinoBridge.ino  (in this repo)

This driver is carefully matched to the EXACT wire protocol of the
ROSArduinoBridge firmware.  Do NOT confuse this with the "diffdrive_arduino"
firmware – they are different projects with different protocols.

No ROS required.  Dependencies: pyserial.

══════════════════════════════════════════════════════════════════════
  Arduino hardware wiring  (from motor_driver.h / encoder_driver.h)
══════════════════════════════════════════════════════════════════════

  L298 H-Bridge motor driver
  ──────────────────────────
  Pin  5  → RIGHT_MOTOR_BACKWARD  (PWM)
  Pin  6  → LEFT_MOTOR_BACKWARD   (PWM)
  Pin  9  → RIGHT_MOTOR_FORWARD   (PWM)
  Pin 10  → LEFT_MOTOR_FORWARD    (PWM)
  Pin 12  → RIGHT_MOTOR_ENABLE    (HIGH = enabled)
  Pin 13  → LEFT_MOTOR_ENABLE     (HIGH = enabled)

  Quadrature encoders (interrupt-driven)
  ───────────────────────────────────────
  Pin  2  (PD2) → LEFT  encoder channel A   ← PCINT2 interrupt
  Pin  3  (PD3) → LEFT  encoder channel B   ← PCINT2 interrupt
  Pin A4  (PC4) → RIGHT encoder channel A   ← PCINT1 interrupt
  Pin A5  (PC5) → RIGHT encoder channel B   ← PCINT1 interrupt

  Serial
  ───────
  Baud rate : 57600
  Terminator: CR  (\\r = 0x0D = ASCII 13)   ← NOT newline!

══════════════════════════════════════════════════════════════════════
  Serial command protocol  (commands.h + ROSArduinoBridge.ino)
══════════════════════════════════════════════════════════════════════

  Command  Send (+ \\r)         Response
  ───────  ──────────────────  ─────────────────────────────────────
  e        "e\\r"               "<left_ticks> <right_ticks>\\r\\n"
  m        "m <l> <r>\\r"      "OK\\r\\n"
           l, r = ticks per PID frame  (see TICKS PER FRAME below)
  o        "o <l> <r>\\r"      "OK\\r\\n"   raw PWM  [-255, 255]
  r        "r\\r"               "OK\\r\\n"   reset encoders + PID
  u        "u Kp:Kd:Ki:Ko\\r"  "OK\\r\\n"   update PID gains
  b        "b\\r"               "57600\\r\\n" get baud rate
  a        "a <pin>\\r"         "<value>\\r\\n" analogRead
  d        "d <pin>\\r"         "<value>\\r\\n" digitalRead
  w        "w <pin> <val>\\r"  "OK\\r\\n"   digitalWrite
  x        "x <pin> <val>\\r"  "OK\\r\\n"   analogWrite
  c        "c <pin> <dir>\\r"  "OK\\r\\n"   pinMode

  TICKS PER FRAME
  ────────────────
  The 'm' command takes TargetTicksPerFrame (integer), NOT ticks/second.
    PID_RATE     = 30 Hz  (defined in ROSArduinoBridge.ino)
    PID_INTERVAL = 1000/30 = 33 ms
    ticks_per_frame = round(ticks_per_second / PID_RATE)

  Example: drive at 0.15 m/s
    ticks/s         = 0.15 / METRES_PER_TICK  ≈ 2487
    ticks/frame     = round(2487 / 30)         ≈ 83
    → send "m 83 83\\r"

  PID default gains  (diff_controller.h)
  ────────────────────────────────────────
    Kp=20  Kd=12  Ki=0  Ko=50

  AUTO-STOP
  ──────────
  Arduino stops motors if no 'm' command arrives within 2000 ms.

══════════════════════════════════════════════════════════════════════
  ENC_COUNTS_PER_REV  (robot_config.py)
══════════════════════════════════════════════════════════════════════
  Value = 3436 (from ros2_control.xacro).
  = encoder_pulses_per_rev × 4 (quadrature) × gear_ratio
  Tune with:  python3 arduino_driver.py [/dev/ttyUSB0]

Usage
─────
    from arduino_driver import ArduinoDriver

    driver = ArduinoDriver()
    driver.start()
    driver.set_speed_ms(0.15, 0.15)
    time.sleep(2)
    driver.stop()
    l, r = driver.read_encoders()
    print(driver.odometry)
    driver.close()
"""

from __future__ import annotations
import math
import threading
import time
from typing import Optional, Tuple

from robot_config import (
    ARDUINO_SERIAL_PORT, ARDUINO_BAUD_RATE, ARDUINO_TIMEOUT_MS,
    ARDUINO_LOOP_RATE_HZ,
    ENC_COUNTS_PER_REV, METRES_PER_TICK,
    WHEEL_RADIUS, WHEEL_SEPARATION,
    MAX_LINEAR_VEL,
)
from diff_drive import RobotState, _wrap_angle


# PID constants matching ROSArduinoBridge.ino defaults
_PID_RATE     = 30                  # Hz  – must match PID_RATE in .ino
_PID_INTERVAL = 1.0 / _PID_RATE    # seconds per PID frame  (≈ 0.03333 s)
_AUTO_STOP_MS = 2000                # ms  – Arduino auto-stop timeout


class ArduinoDriver:
    """
    Communicate with the ROSArduinoBridge firmware over USB serial.

    Compile-time options active in the .ino sketch:
        #define USE_BASE
        #define ARDUINO_ENC_COUNTER
        #define L298_MOTOR_DRIVER
        BAUDRATE = 57600

    Parameters
    ----------
    port         : serial device, e.g. "/dev/ttyUSB0"
    baud_rate    : must match BAUDRATE in .ino  (default 57600)
    loop_rate_hz : how often Python polls encoders & sends commands
    """

    def __init__(
        self,
        port:         str = ARDUINO_SERIAL_PORT,
        baud_rate:    int = ARDUINO_BAUD_RATE,
        loop_rate_hz: int = ARDUINO_LOOP_RATE_HZ,
    ):
        self.port         = port
        self.baud_rate    = baud_rate
        self.loop_rate_hz = loop_rate_hz

        self._ser     = None
        self._lock    = threading.Lock()
        self._thread  = None
        self._running = False

        # Raw encoder accumulators
        self._enc_left:   int = 0
        self._enc_right:  int = 0
        self._prev_left:  int = 0
        self._prev_right: int = 0

        # Odometry integrated from ticks
        self._odom = RobotState()

        # Speed commands in ticks-per-PID-frame (what 'm' command takes)
        self._cmd_tpf_left:  int = 0
        self._cmd_tpf_right: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open serial port and start background control loop."""
        try:
            import serial
        except ImportError:
            raise ImportError("pyserial not installed.  Run:  pip install pyserial")

        import serial as _ser_mod
        self._ser = _ser_mod.Serial(
            self.port,
            self.baud_rate,
            timeout=ARDUINO_TIMEOUT_MS / 1000.0,
        )
        # Wait for Arduino bootloader to finish (DTR toggle triggers reset)
        time.sleep(2.0)
        self._ser.reset_input_buffer()

        self._running = True
        self._thread  = threading.Thread(
            target=self._control_loop, daemon=True, name="arduino-ctrl"
        )
        self._thread.start()

        baud = self._get_baud_rate()
        print(f"[Arduino] Connected on {self.port} @ {self.baud_rate} baud")
        print(f"[Arduino] Firmware baud check: {baud}")
        print(f"[Arduino] ENC_COUNTS_PER_REV = {ENC_COUNTS_PER_REV}  "
              f"({METRES_PER_TICK * 1e6:.2f} µm/tick)")
        print(f"[Arduino] PID_RATE={_PID_RATE} Hz  "
              f"AUTO_STOP={_AUTO_STOP_MS} ms")

    def close(self) -> None:
        """Stop control loop, send stop, close serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._send("m 0 0")
        except Exception:
            pass
        if self._ser and self._ser.is_open:
            self._ser.close()
        print("[Arduino] Connection closed.")

    # ── Velocity commands ─────────────────────────────────────────────────────

    def set_speed_tpf(self, left_tpf: int, right_tpf: int) -> None:
        """
        Set wheel speed in ticks per PID frame ('m' command units).

        1 frame = 1000 ms / PID_RATE = 33 ms at default 30 Hz.
        Values are signed integers (negative = reverse).
        """
        with self._lock:
            self._cmd_tpf_left  = int(left_tpf)
            self._cmd_tpf_right = int(right_tpf)

    def set_speed_tps(self, left_tps: float, right_tps: float) -> None:
        """Set wheel speed in ticks per second. Converts to ticks/frame."""
        self.set_speed_tpf(
            round(left_tps  * _PID_INTERVAL),
            round(right_tps * _PID_INTERVAL),
        )

    def set_speed_ms(self, left_ms: float, right_ms: float) -> None:
        """Set individual wheel speeds in metres per second."""
        self.set_speed_tps(
            left_ms  / METRES_PER_TICK,
            right_ms / METRES_PER_TICK,
        )

    def set_velocity(self, linear_ms: float, angular_rads: float) -> None:
        """
        Set robot velocity as (linear m/s, angular rad/s).
        Differential-drive inverse kinematics converts to wheel speeds.
        """
        v_l = linear_ms - angular_rads * WHEEL_SEPARATION / 2.0
        v_r = linear_ms + angular_rads * WHEEL_SEPARATION / 2.0
        v_l = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, v_l))
        v_r = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, v_r))
        self.set_speed_ms(v_l, v_r)

    def set_raw_pwm(self, left_pwm: int, right_pwm: int) -> None:
        """
        Send raw PWM values [-255, 255] bypassing PID (uses 'o' command).
        Use for motor testing only.
        """
        self._send(f"o {int(left_pwm)} {int(right_pwm)}")

    def stop(self) -> None:
        """Stop both motors (sends 'm 0 0', which also resets PID)."""
        self.set_speed_tpf(0, 0)

    # ── Encoders / odometry ───────────────────────────────────────────────────

    def read_encoders(self) -> Tuple[int, int]:
        """Return latest (left, right) raw encoder tick counts."""
        with self._lock:
            return self._enc_left, self._enc_right

    def reset_encoders(self) -> None:
        """Send 'r' to Arduino (resets encoders + PID) and clear local state."""
        resp = self._send("r")
        if resp and "OK" in resp:
            with self._lock:
                self._enc_left   = 0
                self._enc_right  = 0
                self._prev_left  = 0
                self._prev_right = 0
                self._odom       = RobotState()
            print("[Arduino] Encoders + PID reset.")
        else:
            print(f"[Arduino] Reset failed – response: {resp!r}")

    @property
    def odometry(self) -> RobotState:
        """Pose integrated from encoder ticks (updated in background thread)."""
        with self._lock:
            return RobotState(self._odom.x, self._odom.y, self._odom.theta)

    def metres_to_tpf(self, metres_per_sec: float) -> int:
        """Convert m/s wheel speed to ticks-per-frame for the 'm' command."""
        return round(metres_per_sec / METRES_PER_TICK * _PID_INTERVAL)

    def ticks_to_metres(self, ticks: int) -> float:
        return ticks * METRES_PER_TICK

    # ── PID tuning ────────────────────────────────────────────────────────────

    def update_pid(self, kp: int, kd: int, ki: int, ko: int) -> bool:
        """
        Send 'u' command to update PID gains on the Arduino.

        Default values from diff_controller.h: Kp=20, Kd=12, Ki=0, Ko=50.
        """
        resp = self._send(f"u {kp}:{kd}:{ki}:{ko}")
        ok = resp is not None and "OK" in resp
        print(f"[Arduino] PID {'updated' if ok else 'FAILED'}: "
              f"Kp={kp} Kd={kd} Ki={ki} Ko={ko}")
        return ok

    # ── GPIO passthrough ──────────────────────────────────────────────────────

    def analog_read(self, pin: int) -> Optional[int]:
        resp = self._send(f"a {pin}")
        try: return int(resp.strip())
        except (ValueError, AttributeError): return None

    def digital_read(self, pin: int) -> Optional[int]:
        resp = self._send(f"d {pin}")
        try: return int(resp.strip())
        except (ValueError, AttributeError): return None

    def digital_write(self, pin: int, value: int) -> bool:
        return "OK" in (self._send(f"w {pin} {value}") or "")

    def analog_write(self, pin: int, value: int) -> bool:
        return "OK" in (self._send(f"x {pin} {value}") or "")

    def pin_mode(self, pin: int, direction: int) -> bool:
        """direction: 0=INPUT, 1=OUTPUT"""
        return "OK" in (self._send(f"c {pin} {direction}") or "")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_baud_rate(self) -> str:
        resp = self._send("b")
        return resp.strip() if resp else "unknown"

    def _control_loop(self) -> None:
        dt = 1.0 / self.loop_rate_hz
        while self._running:
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as e:
                print(f"[Arduino] Control loop error: {e}")
            elapsed = time.monotonic() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _tick(self) -> None:
        # Send motor speed command
        with self._lock:
            cmd_l = self._cmd_tpf_left
            cmd_r = self._cmd_tpf_right
        self._send(f"m {cmd_l} {cmd_r}")

        # Read encoders
        raw = self._send("e")
        if not raw:
            return
        parts = raw.strip().split()
        if len(parts) < 2:
            return
        try:
            new_left  = int(parts[0])
            new_right = int(parts[1])
        except ValueError:
            return

        # Integrate odometry from tick deltas
        with self._lock:
            dl = new_left  - self._prev_left
            dr = new_right - self._prev_right
            self._enc_left   = new_left
            self._enc_right  = new_right
            self._prev_left  = new_left
            self._prev_right = new_right

            dist_l = dl * METRES_PER_TICK
            dist_r = dr * METRES_PER_TICK
            dist   = (dist_l + dist_r) / 2.0
            dtheta = (dist_r - dist_l) / WHEEL_SEPARATION

            # Mid-point Euler integration
            self._odom.x     += dist * math.cos(self._odom.theta + dtheta / 2.0)
            self._odom.y     += dist * math.sin(self._odom.theta + dtheta / 2.0)
            self._odom.theta  = _wrap_angle(self._odom.theta + dtheta)

    def _send(self, cmd: str) -> Optional[str]:
        """
        Send a command + CR terminator and read one response line.

        IMPORTANT: ROSArduinoBridge terminates commands with CR (\\r, chr==13),
        NOT with newline (\\n).  See the loop() function in ROSArduinoBridge.ino:
            if (chr == 13) { ... runCommand(); }
        """
        if not self._ser or not self._ser.is_open:
            return None
        try:
            self._ser.reset_input_buffer()
            self._ser.write((cmd + "\r").encode())   # ← CR terminator
            return self._ser.readline().decode(errors='replace')
        except Exception as e:
            print(f"[Arduino] Serial error on '{cmd}': {e}")
            return None


# ── High-level robot interface ────────────────────────────────────────────────

class ArduinoRobot:
    """
    Convenience wrapper combining ArduinoDriver with a clean API.

    Usage
    ─────
        robot = ArduinoRobot()
        robot.start()
        robot.set_velocity(0.15, 0.0)
        time.sleep(2)
        robot.stop()
        print(robot.pose)
        robot.close()
    """

    def __init__(self, port: str = ARDUINO_SERIAL_PORT):
        self.driver = ArduinoDriver(port=port)

    def start(self)  -> None: self.driver.start()
    def close(self)  -> None: self.driver.close()
    def stop(self)   -> None: self.driver.stop()

    def set_velocity(self, linear: float, angular: float) -> None:
        self.driver.set_velocity(linear, angular)

    @property
    def pose(self) -> RobotState:
        return self.driver.odometry

    def read_encoders(self) -> Tuple[int, int]:
        return self.driver.read_encoders()

    def reset_encoders(self) -> None:
        self.driver.reset_encoders()

    def update_pid(self, kp: int = 20, kd: int = 12,
                   ki: int = 0, ko: int = 50) -> bool:
        return self.driver.update_pid(kp, kd, ki, ko)


# ── Encoder calibration wizard ────────────────────────────────────────────────

def run_encoder_tuning(port: str = ARDUINO_SERIAL_PORT) -> None:
    """
    Measure the correct ENC_COUNTS_PER_REV for your motors.

    Run standalone:
        python3 arduino_driver.py [/dev/ttyUSB0]
    """
    print("\n══════════════════════════════════════════════════")
    print("  ROSArduinoBridge – Encoder Calibration Wizard")
    print("══════════════════════════════════════════════════")
    print(f"\n  Firmware   : ROSArduinoBridge (L298 + ARDUINO_ENC_COUNTER)")
    print(f"  Encoder pins:  LEFT  A=D2(PD2)  B=D3(PD3)")
    print(f"                 RIGHT A=A4(PC4)  B=A5(PC5)")
    print(f"\n  Current ENC_COUNTS_PER_REV = {ENC_COUNTS_PER_REV}")
    print(f"  Current METRES_PER_TICK    = {METRES_PER_TICK:.8f} m")
    print(f"  PID_RATE                   = {_PID_RATE} Hz")
    print(f"  PID_INTERVAL               = {_PID_INTERVAL*1000:.2f} ms\n")

    driver = ArduinoDriver(port=port)
    driver.start()
    driver.reset_encoders()

    default_tpf = 10
    approx_ms   = default_tpf * _PID_RATE * METRES_PER_TICK
    tpf_str = input(
        f"  Test speed in ticks/frame [{default_tpf}]  "
        f"(≈ {approx_ms:.3f} m/s): "
    ).strip() or str(default_tpf)
    tpf = int(tpf_str)

    duration = float(input("  Drive duration in seconds [3.0]: ").strip() or "3.0")

    actual_ms = tpf * _PID_RATE * METRES_PER_TICK
    print(f"\n  Driving at {tpf} ticks/frame ≈ {actual_ms:.3f} m/s "
          f"for {duration} s …")

    driver.set_speed_tpf(tpf, tpf)
    time.sleep(duration)
    driver.stop()
    time.sleep(0.5)

    l, r = driver.read_encoders()
    avg  = (l + r) / 2.0
    print(f"\n  Encoder counts:  left={l}  right={r}  avg={avg:.0f} ticks")

    dist_str = input(
        "  Measure the actual distance the robot travelled (metres): "
    ).strip()
    if not dist_str:
        print("  (skipped)")
        driver.close()
        return

    actual_m   = float(dist_str)
    wheel_circ = 2 * math.pi * WHEEL_RADIUS
    new_cpr    = avg / actual_m * wheel_circ

    print(f"\n  ── Result ─────────────────────────────────────────")
    print(f"  Measured distance   : {actual_m:.4f} m")
    print(f"  Wheel circumference : {wheel_circ:.5f} m")
    print(f"  Calculated CPR      : {new_cpr:.1f}  → integer: {round(new_cpr)}")
    print(f"  New METRES_PER_TICK : {wheel_circ / round(new_cpr):.8f} m")
    print(f"\n  ► Edit  python_main_script/robot_config.py :")
    print(f"      ENC_COUNTS_PER_REV = {round(new_cpr)}")
    print(f"  ───────────────────────────────────────────────────\n")

    driver.close()


# ── PID tuning helper ─────────────────────────────────────────────────────────

def run_pid_tuning(port: str = ARDUINO_SERIAL_PORT) -> None:
    """
    Interactive PID gain tuning session.
    Default gains from diff_controller.h: Kp=20, Kd=12, Ki=0, Ko=50.

    Run standalone:
        python3 arduino_driver.py --pid [/dev/ttyUSB0]
    """
    print("\n══════════════════════════════════════════════")
    print("  ROSArduinoBridge – PID Tuning")
    print("══════════════════════════════════════════════")
    print("  Defaults (diff_controller.h): Kp=20  Kd=12  Ki=0  Ko=50\n")

    driver = ArduinoDriver(port=port)
    driver.start()

    while True:
        raw = input("  Enter Kp Kd Ki Ko  (or 'q' to quit): ").strip()
        if raw.lower() == 'q':
            break
        parts = (raw or "20 12 0 50").split()
        if len(parts) != 4:
            print("  Need exactly 4 integers.")
            continue
        try:
            kp, kd, ki, ko = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            print("  Values must be integers.")
            continue
        driver.update_pid(kp, kd, ki, ko)

        tpf = int(input("  Test ticks/frame [15]: ").strip() or "15")
        dur = float(input("  Drive duration seconds [2]: ").strip() or "2")
        driver.reset_encoders()
        driver.set_speed_tpf(tpf, tpf)
        time.sleep(dur)
        driver.stop()
        time.sleep(0.3)
        l, r = driver.read_encoders()
        print(f"  → Left={l}  Right={r}  diff={abs(l-r)} ticks")
        print("    (smaller diff = straighter driving)\n")

    driver.close()


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    port = ARDUINO_SERIAL_PORT
    mode = 'calibrate'
    for a in args:
        if a == '--pid':
            mode = 'pid'
        elif not a.startswith('--'):
            port = a
    if mode == 'pid':
        run_pid_tuning(port)
    else:
        run_encoder_tuning(port)

