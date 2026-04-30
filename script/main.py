#!/usr/bin/env python3
"""
VacuumBot Control Server
Raspberry Pi backend for LD06 lidar + Arduino differential drive vacuum robot.
Provides WebSocket real-time communication and HTTP API for the web interface.
"""

import asyncio
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

import serial
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sock import Sock

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/vacuumbot.log", mode="a"),
    ],
)
log = logging.getLogger("vacuumbot")

# ── Configuration ─────────────────────────────────────────────────────────────
ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "/dev/ttyUSB0")
LIDAR_PORT   = os.environ.get("LIDAR_PORT",   "/dev/ttyUSB1")
ARDUINO_BAUD = 57600
LIDAR_BAUD   = 230400
WEB_PORT     = int(os.environ.get("WEB_PORT", 5000))

# Robot physical parameters
WHEEL_DIAMETER_M  = 0.065   # 65 mm wheels
WHEEL_BASE_M      = 0.150   # 150 mm track width
ENCODER_TICKS_REV = 1440    # ticks per full revolution
TICKS_PER_METER   = int((1.0 / (math.pi * WHEEL_DIAMETER_M)) * ENCODER_TICKS_REV)
TICKS_PER_FRAME   = 30      # PID runs at 30 Hz; base speed in ticks/frame

# Map parameters
MAP_RESOLUTION = 0.05       # meters per cell (5 cm)
MAP_SIZE_M     = 10.0       # 10 × 10 metre map
MAP_CELLS      = int(MAP_SIZE_M / MAP_RESOLUTION)   # 200 × 200
MAP_ORIGIN_X   = MAP_CELLS // 2
MAP_ORIGIN_Y   = MAP_CELLS // 2

# Cleaning parameters
ZIGZAG_STRIP_WIDTH_M = 0.20   # width of each cleaning pass

MAP_FILE = Path("/var/lib/vacuumbot/map.json")
MAP_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Shared robot state ────────────────────────────────────────────────────────
class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        # Pose (meters, radians)
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        # Raw encoder counts
        self.enc_left  = 0
        self.enc_right = 0
        # Lidar scan (list of {angle_deg, distance_m})
        self.scan = []
        # Occupancy grid: 0=unknown, 1=free, 2=occupied
        self.grid = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
        # Cleaned cells
        self.cleaned = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
        # Robot mode
        self.mode = "idle"          # idle | mapping | navigating | cleaning
        self.map_saved   = False
        self.map_locked  = False    # True once map saved; prevents overwrite
        # Navigation goal
        self.goal_x = None
        self.goal_y = None
        # Cleaning plan (list of waypoints)
        self.clean_plan  = []
        self.clean_index = 0
        # Connection health
        self.arduino_ok = False
        self.lidar_ok   = False

state = RobotState()

# ── Arduino Serial Interface ──────────────────────────────────────────────────
class ArduinoDriver:
    """Thread-safe serial interface to the ROSArduinoBridge firmware."""

    def __init__(self, port: str, baud: int):
        self.port  = port
        self.baud  = baud
        self._ser  = None
        self._lock = threading.Lock()
        self._prev_left  = 0
        self._prev_right = 0
        self._connect()

    def _connect(self):
        for attempt in range(10):
            try:
                self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
                time.sleep(2)          # Arduino resets on connect
                self._ser.flushInput()
                log.info("Arduino connected on %s", self.port)
                state.arduino_ok = True
                return
            except serial.SerialException as e:
                log.warning("Arduino connect attempt %d failed: %s", attempt + 1, e)
                time.sleep(2)
        log.error("Could not connect to Arduino on %s", self.port)

    def _cmd(self, command: str) -> str:
        """Send a command and return the response line."""
        if not self._ser or not self._ser.is_open:
            return ""
        with self._lock:
            try:
                self._ser.write((command + "\r").encode())
                resp = self._ser.readline().decode(errors="ignore").strip()
                return resp
            except serial.SerialException as e:
                log.error("Arduino serial error: %s", e)
                state.arduino_ok = False
                return ""

    def set_motor_speeds(self, left_tpf: int, right_tpf: int):
        """Set motor speeds in ticks-per-frame."""
        left_tpf  = max(-TICKS_PER_FRAME * 5, min(TICKS_PER_FRAME * 5, left_tpf))
        right_tpf = max(-TICKS_PER_FRAME * 5, min(TICKS_PER_FRAME * 5, right_tpf))
        self._cmd(f"m {left_tpf} {right_tpf}")

    def stop(self):
        self._cmd("m 0 0")

    def read_encoders(self):
        """Return (left, right) encoder counts."""
        resp = self._cmd("e")
        try:
            parts = resp.split()
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return self._prev_left, self._prev_right

    def reset_encoders(self):
        self._cmd("r")
        self._prev_left  = 0
        self._prev_right = 0

    def update_odometry(self):
        """Read encoders and integrate odometry into robot state."""
        left, right = self.read_encoders()
        dl = left  - self._prev_left
        dr = right - self._prev_right
        self._prev_left  = left
        self._prev_right = right

        dist_left  = dl / TICKS_PER_METER
        dist_right = dr / TICKS_PER_METER
        dist       = (dist_left + dist_right) / 2.0
        d_yaw      = (dist_right - dist_left) / WHEEL_BASE_M

        with state.lock:
            state.enc_left  = left
            state.enc_right = right
            state.yaw += d_yaw
            state.x   += dist * math.cos(state.yaw)
            state.y   += dist * math.sin(state.yaw)
            # Wrap yaw to [-π, π]
            state.yaw = math.atan2(math.sin(state.yaw), math.cos(state.yaw))

    def close(self):
        if self._ser and self._ser.is_open:
            self.stop()
            self._ser.close()


# ── LD06 Lidar Driver ─────────────────────────────────────────────────────────
class LD06Driver:
    """
    Reads the LD06 LIDAR serial stream (230400 baud).
    Packet format: 0x54 header, 0x2C (12 points per packet), speed(2),
    start_angle(2), 12×{dist(2),intensity(1)}, end_angle(2), timestamp(2), crc(1).
    Total: 47 bytes per packet.
    """
    HEADER = 0x54
    PKT_LEN = 47

    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self._ser = None
        self._thread = None
        self._running = False
        self._connect()

    def _connect(self):
        for attempt in range(10):
            try:
                self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
                self._ser.flushInput()
                log.info("LD06 connected on %s", self.port)
                state.lidar_ok = True
                return
            except serial.SerialException as e:
                log.warning("LD06 connect attempt %d failed: %s", attempt + 1, e)
                time.sleep(2)
        log.error("Could not connect to LD06 on %s", self.port)

    def _crc8(self, data: bytes) -> int:
        crc_table = [
            0x00,0x4d,0x9a,0xd7,0x79,0x34,0xe3,0xae,
            0xf2,0xbf,0x68,0x25,0x8b,0xc6,0x11,0x5c,
            0xa9,0xe4,0x33,0x7e,0xd0,0x9d,0x4a,0x07,
            0x5b,0x16,0xc1,0x8c,0x22,0x6f,0xb8,0xf5,
            0x1f,0x52,0x85,0xc8,0x66,0x2b,0xfc,0xb1,
            0xed,0xa0,0x77,0x3a,0x94,0xd9,0x0e,0x43,
            0xb6,0xfb,0x2c,0x61,0xcf,0x82,0x55,0x18,
            0x44,0x09,0xde,0x93,0x3d,0x70,0xa7,0xea,
            0x3e,0x73,0xa4,0xe9,0x47,0x0a,0xdd,0x90,
            0xcc,0x81,0x56,0x1b,0xb5,0xf8,0x2f,0x62,
            0x97,0xda,0x0d,0x40,0xee,0xa3,0x74,0x39,
            0x65,0x28,0xff,0xb2,0x1c,0x51,0x86,0xcb,
            0x21,0x6c,0xbb,0xf6,0x58,0x15,0xc2,0x8f,
            0xd3,0x9e,0x49,0x04,0xaa,0xe7,0x30,0x7d,
            0x88,0xc5,0x12,0x5f,0xf1,0xbc,0x6b,0x26,
            0x7a,0x37,0xe0,0xad,0x03,0x4e,0x99,0xd4,
            0x7c,0x31,0xe6,0xab,0x05,0x48,0x9f,0xd2,
            0x8e,0xc3,0x14,0x59,0xf7,0xba,0x6d,0x20,
            0xd5,0x98,0x4f,0x02,0xac,0xe1,0x36,0x7b,
            0x27,0x6a,0xbd,0xf0,0x5e,0x13,0xc4,0x89,
            0x63,0x2e,0xf9,0xb4,0x1a,0x57,0x80,0xcd,
            0x91,0xdc,0x0b,0x46,0xe8,0xa5,0x72,0x3f,
            0xca,0x87,0x50,0x1d,0xb3,0xfe,0x29,0x64,
            0x38,0x75,0xa2,0xef,0x41,0x0c,0xdb,0x96,
            0x42,0x0f,0xd8,0x95,0x3b,0x76,0xa1,0xec,
            0xb0,0xfd,0x2a,0x67,0xc9,0x84,0x53,0x1e,
            0xeb,0xa6,0x71,0x3c,0x92,0xdf,0x08,0x45,
            0x19,0x54,0x83,0xce,0x60,0x2d,0xfa,0xb7,
            0x5d,0x10,0xc7,0x8a,0x24,0x69,0xbe,0xf3,
            0xaf,0xe2,0x35,0x78,0xd6,0x9b,0x4c,0x01,
            0xf4,0xb9,0x6e,0x23,0x8d,0xc0,0x17,0x5a,
            0x06,0x4b,0x9c,0xd1,0x7f,0x32,0xe5,0xa8,
        ]
        crc = 0
        for b in data:
            crc = crc_table[(crc ^ b) & 0xFF]
        return crc

    def _parse_packet(self, pkt: bytes) -> list:
        """Parse one 47-byte LD06 packet. Returns list of (angle_deg, dist_m)."""
        if len(pkt) != self.PKT_LEN:
            return []
        if pkt[0] != self.HEADER or pkt[1] != 0x2C:
            return []
        # CRC check (first 46 bytes)
        if self._crc8(pkt[:46]) != pkt[46]:
            return []

        start_angle = (pkt[4] | (pkt[5] << 8)) / 100.0   # degrees
        end_angle   = (pkt[40] | (pkt[41] << 8)) / 100.0  # degrees

        # Handle angle wraparound
        if end_angle < start_angle:
            end_angle += 360.0
        angle_step = (end_angle - start_angle) / 11.0      # 12 points

        points = []
        for i in range(12):
            base = 6 + i * 3
            dist_raw  = pkt[base] | (pkt[base + 1] << 8)
            # intensity = pkt[base + 2]  # available if needed
            dist_m = dist_raw / 1000.0
            angle  = (start_angle + i * angle_step) % 360.0
            if 0.05 < dist_m < 8.0:   # filter noise / max range
                points.append((angle, dist_m))
        return points

    def _reader_thread(self):
        buf = bytearray()
        while self._running:
            if not self._ser or not self._ser.is_open:
                time.sleep(0.1)
                continue
            try:
                chunk = self._ser.read(self._ser.in_waiting or 1)
                if not chunk:
                    continue
                buf.extend(chunk)
                # Find and consume complete packets
                while len(buf) >= self.PKT_LEN:
                    idx = buf.find(bytes([self.HEADER]))
                    if idx == -1:
                        buf.clear()
                        break
                    if idx > 0:
                        del buf[:idx]
                    if len(buf) < self.PKT_LEN:
                        break
                    pkt    = bytes(buf[:self.PKT_LEN])
                    points = self._parse_packet(pkt)
                    if points:
                        with state.lock:
                            state.scan = points
                        # Update map in mapping mode
                        if state.mode in ("mapping", "cleaning"):
                            self._update_map(points)
                    del buf[:self.PKT_LEN]
            except serial.SerialException as e:
                log.error("LD06 read error: %s", e)
                state.lidar_ok = False
                time.sleep(0.5)

    def _update_map(self, points):
        """Bresenham ray-casting to update occupancy grid."""
        with state.lock:
            rx = state.x
            ry = state.y
            ryaw = state.yaw

        ox = int(rx / MAP_RESOLUTION) + MAP_ORIGIN_X
        oy = int(ry / MAP_RESOLUTION) + MAP_ORIGIN_Y

        for angle_deg, dist_m in points:
            world_angle = math.radians(angle_deg) + ryaw
            ex = int((rx + dist_m * math.cos(world_angle)) / MAP_RESOLUTION) + MAP_ORIGIN_X
            ey = int((ry + dist_m * math.sin(world_angle)) / MAP_RESOLUTION) + MAP_ORIGIN_Y

            # Bresenham line: free cells along the ray
            for cx, cy in _bresenham(ox, oy, ex, ey):
                if 0 <= cx < MAP_CELLS and 0 <= cy < MAP_CELLS:
                    with state.lock:
                        if state.grid[cy, cx] != 2:
                            state.grid[cy, cx] = 1  # free

            # Endpoint: occupied
            if 0 <= ex < MAP_CELLS and 0 <= ey < MAP_CELLS:
                with state.lock:
                    state.grid[ey, ex] = 2

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader_thread, daemon=True, name="ld06-reader")
        self._thread.start()

    def close(self):
        self._running = False
        if self._ser and self._ser.is_open:
            self._ser.close()


def _bresenham(x0, y0, x1, y1):
    """Yield integer (x, y) cells along the line from (x0,y0) to (x1,y1)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 < dx:
            err += dx
            y0  += sy
    return cells


# ── Navigation Controller ─────────────────────────────────────────────────────
class Navigator:
    """
    Simple proportional heading + distance controller for point-to-point navigation.
    Runs in its own thread, driven by a waypoint queue.
    """
    GOAL_TOLERANCE    = 0.08   # metres
    HEADING_TOLERANCE = 0.05   # radians (~3°)
    SPEED_TICKS       = 20     # ticks/frame cruise speed
    TURN_TICKS        = 10     # ticks/frame turn speed

    def __init__(self, arduino: ArduinoDriver):
        self.arduino  = arduino
        self._running = False
        self._thread  = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="navigator")
        self._thread.start()

    def _loop(self):
        while self._running:
            time.sleep(0.05)   # 20 Hz

            with state.lock:
                mode = state.mode
                gx   = state.goal_x
                gy   = state.goal_y
                rx   = state.x
                ry   = state.y
                ryaw = state.yaw

            if mode not in ("navigating", "cleaning") or gx is None:
                continue

            dx = gx - rx
            dy = gy - ry
            dist = math.hypot(dx, dy)

            if dist < self.GOAL_TOLERANCE:
                self.arduino.stop()
                self._on_goal_reached()
                continue

            target_yaw = math.atan2(dy, dx)
            yaw_err    = math.atan2(math.sin(target_yaw - ryaw),
                                    math.cos(target_yaw - ryaw))

            if abs(yaw_err) > self.HEADING_TOLERANCE:
                # Pure rotation
                t = self.TURN_TICKS
                if yaw_err > 0:
                    self.arduino.set_motor_speeds(-t, t)
                else:
                    self.arduino.set_motor_speeds(t, -t)
            else:
                # Forward with gentle correction
                correction = int(yaw_err * 15)
                left  = self.SPEED_TICKS - correction
                right = self.SPEED_TICKS + correction
                left  = max(-self.SPEED_TICKS * 2, min(self.SPEED_TICKS * 2, left))
                right = max(-self.SPEED_TICKS * 2, min(self.SPEED_TICKS * 2, right))
                self.arduino.set_motor_speeds(left, right)

    def _on_goal_reached(self):
        with state.lock:
            mode = state.mode
            idx  = state.clean_index
            plan = state.clean_plan

        if mode == "navigating":
            with state.lock:
                state.mode   = "idle"
                state.goal_x = None
                state.goal_y = None
            log.info("Navigation goal reached")

        elif mode == "cleaning":
            next_idx = idx + 1
            if next_idx < len(plan):
                with state.lock:
                    state.clean_index = next_idx
                    state.goal_x      = plan[next_idx][0]
                    state.goal_y      = plan[next_idx][1]
                # Mark cleaned area around current position
                self._mark_cleaned()
            else:
                with state.lock:
                    state.mode        = "idle"
                    state.goal_x      = None
                    state.goal_y      = None
                    state.clean_index = 0
                log.info("Cleaning complete!")
                self.arduino.stop()

    def _mark_cleaned(self):
        with state.lock:
            cx = int(state.x / MAP_RESOLUTION) + MAP_ORIGIN_X
            cy = int(state.y / MAP_RESOLUTION) + MAP_ORIGIN_Y
            r  = max(1, int((ZIGZAG_STRIP_WIDTH_M / 2) / MAP_RESOLUTION))
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < MAP_CELLS and 0 <= ny < MAP_CELLS:
                        state.cleaned[ny, nx] = 1

    def stop(self):
        self._running = False


# ── Zigzag Cleaning Planner ───────────────────────────────────────────────────
def generate_zigzag_plan() -> list:
    """
    Generate a boustrophedon (zigzag) cleaning path over all known free cells.
    Returns a list of (x_m, y_m) waypoints.
    """
    with state.lock:
        grid = state.grid.copy()

    strip_cells = max(1, int(ZIGZAG_STRIP_WIDTH_M / MAP_RESOLUTION))
    waypoints = []

    # Find bounding box of free space
    free = np.argwhere(grid == 1)
    if len(free) == 0:
        return []

    min_col = int(free[:, 1].min())
    max_col = int(free[:, 1].max())
    min_row = int(free[:, 0].min())
    max_row = int(free[:, 0].max())

    col = min_col
    direction = 1   # 1 = top-to-bottom, -1 = bottom-to-top

    while col <= max_col:
        if direction == 1:
            row_range = range(min_row, max_row + 1)
        else:
            row_range = range(max_row, min_row - 1, -1)

        strip_wps = []
        for row in row_range:
            if grid[row, col] == 1:
                wx = (col - MAP_ORIGIN_X) * MAP_RESOLUTION
                wy = (row - MAP_ORIGIN_Y) * MAP_RESOLUTION
                strip_wps.append((wx, wy))

        if strip_wps:
            waypoints.extend(strip_wps[::3])   # sub-sample every 3 cells

        col       += strip_cells
        direction *= -1

    log.info("Generated zigzag plan with %d waypoints", len(waypoints))
    return waypoints


# ── Map persistence ───────────────────────────────────────────────────────────
def save_map():
    with state.lock:
        data = {
            "grid":    state.grid.tolist(),
            "cleaned": state.cleaned.tolist(),
            "pose":    {"x": state.x, "y": state.y, "yaw": state.yaw},
        }
    with open(MAP_FILE, "w") as f:
        json.dump(data, f)
    log.info("Map saved to %s", MAP_FILE)


def load_map():
    if not MAP_FILE.exists():
        return False
    try:
        with open(MAP_FILE) as f:
            data = json.load(f)
        with state.lock:
            state.grid    = np.array(data["grid"], dtype=np.uint8)
            state.cleaned = np.array(data["cleaned"], dtype=np.uint8)
            state.map_saved  = True
            state.map_locked = True
        log.info("Map loaded from %s", MAP_FILE)
        return True
    except Exception as e:
        log.error("Failed to load map: %s", e)
        return False


# ── Odometry polling thread ───────────────────────────────────────────────────
def odometry_thread(arduino: ArduinoDriver):
    while True:
        arduino.update_odometry()
        # Mark robot's current position as cleaned during cleaning mode
        with state.lock:
            if state.mode == "cleaning":
                cx = int(state.x / MAP_RESOLUTION) + MAP_ORIGIN_X
                cy = int(state.y / MAP_RESOLUTION) + MAP_ORIGIN_Y
                if 0 <= cx < MAP_CELLS and 0 <= cy < MAP_CELLS:
                    state.cleaned[cy, cx] = 1
        time.sleep(0.033)  # ~30 Hz matches PID loop


# ── WebSocket broadcast thread ────────────────────────────────────────────────
_ws_clients: set = set()
_ws_lock = threading.Lock()

def broadcast_state():
    """Periodically build and broadcast robot state to all connected WebSocket clients."""
    while True:
        time.sleep(0.1)   # 10 Hz updates
        try:
            with state.lock:
                scan_data = state.scan[::]
                payload = {
                    "type":      "state",
                    "x":         round(state.x, 4),
                    "y":         round(state.y, 4),
                    "yaw":       round(state.yaw, 4),
                    "mode":      state.mode,
                    "map_saved": state.map_saved,
                    "arduino_ok": state.arduino_ok,
                    "lidar_ok":   state.lidar_ok,
                    "scan": [
                        {"a": round(a, 2), "d": round(d, 4)}
                        for a, d in scan_data[:180:3]   # downsample
                    ],
                }

            msg = json.dumps(payload)
            dead = set()
            with _ws_lock:
                clients = set(_ws_clients)
            for ws in clients:
                try:
                    ws.send(msg)
                except Exception:
                    dead.add(ws)
            if dead:
                with _ws_lock:
                    _ws_clients -= dead
        except Exception as e:
            log.debug("Broadcast error: %s", e)


def broadcast_map():
    """Send full map to all clients (lower frequency - on demand or every 2 s)."""
    while True:
        time.sleep(2.0)
        try:
            with state.lock:
                grid_flat    = state.grid.flatten().tolist()
                cleaned_flat = state.cleaned.flatten().tolist()

            payload = json.dumps({
                "type":         "map",
                "width":        MAP_CELLS,
                "height":       MAP_CELLS,
                "resolution":   MAP_RESOLUTION,
                "origin_x":     MAP_ORIGIN_X,
                "origin_y":     MAP_ORIGIN_Y,
                "grid":         grid_flat,
                "cleaned":      cleaned_flat,
            })

            dead = set()
            with _ws_lock:
                clients = set(_ws_clients)
            for ws in clients:
                try:
                    ws.send(payload)
                except Exception:
                    dead.add(ws)
            if dead:
                with _ws_lock:
                    _ws_clients -= dead
        except Exception as e:
            log.debug("Map broadcast error: %s", e)


# ── Flask Application ─────────────────────────────────────────────────────────
app  = Flask(__name__, static_folder="../frontend")
CORS(app)
sock = Sock(app)

@sock.route("/ws")
def websocket(ws):
    with _ws_lock:
        _ws_clients.add(ws)
    log.info("WebSocket client connected (%d total)", len(_ws_clients))
    # Send immediate map on connect
    try:
        with state.lock:
            grid_flat    = state.grid.flatten().tolist()
            cleaned_flat = state.cleaned.flatten().tolist()
        ws.send(json.dumps({
            "type":       "map",
            "width":      MAP_CELLS,
            "height":     MAP_CELLS,
            "resolution": MAP_RESOLUTION,
            "origin_x":   MAP_ORIGIN_X,
            "origin_y":   MAP_ORIGIN_Y,
            "grid":       grid_flat,
            "cleaned":    cleaned_flat,
        }))
    except Exception:
        pass

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            _handle_ws_message(ws, msg)
    except Exception as e:
        log.debug("WebSocket closed: %s", e)
    finally:
        with _ws_lock:
            _ws_clients.discard(ws)
        log.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))


def _handle_ws_message(ws, raw: str):
    """Handle incoming WebSocket commands from the web UI."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return

    cmd = msg.get("cmd")

    if cmd == "manual":
        # {"cmd":"manual", "left":10, "right":10}
        if state.mode not in ("navigating", "cleaning"):
            left  = int(msg.get("left",  0))
            right = int(msg.get("right", 0))
            arduino.set_motor_speeds(left, right)

    elif cmd == "stop":
        with state.lock:
            state.mode   = "idle"
            state.goal_x = None
            state.goal_y = None
            state.clean_plan  = []
            state.clean_index = 0
        arduino.stop()

    elif cmd == "start_mapping":
        with state.lock:
            if not state.map_locked:
                state.mode = "mapping"
                state.grid[:] = 0
                state.cleaned[:] = 0

    elif cmd == "stop_mapping":
        with state.lock:
            state.mode      = "idle"
            state.map_saved = True
            state.map_locked = True
        save_map()

    elif cmd == "navigate":
        # {"cmd":"navigate", "gx": 1.5, "gy": -0.3}
        gx = float(msg.get("gx", 0))
        gy = float(msg.get("gy", 0))
        with state.lock:
            state.goal_x = gx
            state.goal_y = gy
            state.mode   = "navigating"
        log.info("Navigate to (%.2f, %.2f)", gx, gy)

    elif cmd == "start_cleaning":
        plan = generate_zigzag_plan()
        if not plan:
            return
        with state.lock:
            state.clean_plan  = plan
            state.clean_index = 0
            state.goal_x      = plan[0][0]
            state.goal_y      = plan[0][1]
            state.mode        = "cleaning"
        log.info("Cleaning started, %d waypoints", len(plan))

    elif cmd == "clear_map":
        with state.lock:
            state.grid[:]    = 0
            state.cleaned[:] = 0
            state.map_saved  = False
            state.map_locked = False
            state.mode       = "idle"
        if MAP_FILE.exists():
            MAP_FILE.unlink()

    elif cmd == "reset_pose":
        with state.lock:
            state.x   = 0.0
            state.y   = 0.0
            state.yaw = 0.0
        arduino.reset_encoders()


# ── HTTP REST endpoints ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/status")
def api_status():
    with state.lock:
        return jsonify({
            "x":    round(state.x,   4),
            "y":    round(state.y,   4),
            "yaw":  round(state.yaw, 4),
            "mode": state.mode,
            "map_saved":   state.map_saved,
            "map_locked":  state.map_locked,
            "arduino_ok":  state.arduino_ok,
            "lidar_ok":    state.lidar_ok,
        })

@app.route("/api/map")
def api_map():
    with state.lock:
        return jsonify({
            "width":      MAP_CELLS,
            "height":     MAP_CELLS,
            "resolution": MAP_RESOLUTION,
            "origin_x":   MAP_ORIGIN_X,
            "origin_y":   MAP_ORIGIN_Y,
            "grid":       state.grid.flatten().tolist(),
            "cleaned":    state.cleaned.flatten().tolist(),
        })


# ── Entry point ───────────────────────────────────────────────────────────────
arduino   = None
lidar     = None
navigator = None

def shutdown(sig, frame):
    log.info("Shutting down...")
    if navigator:
        navigator.stop()
    if arduino:
        arduino.stop()
        arduino.close()
    if lidar:
        lidar.close()
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == "__main__":
    log.info("Starting VacuumBot server...")

    arduino   = ArduinoDriver(ARDUINO_PORT, ARDUINO_BAUD)
    lidar     = LD06Driver(LIDAR_PORT, LIDAR_BAUD)
    navigator = Navigator(arduino)

    # Load saved map if it exists
    load_map()

    # Start background threads
    lidar.start()
    navigator.start()

    threading.Thread(target=odometry_thread, args=(arduino,), daemon=True, name="odometry").start()
    threading.Thread(target=broadcast_state, daemon=True, name="broadcast-state").start()
    threading.Thread(target=broadcast_map,   daemon=True, name="broadcast-map").start()

    log.info("Web interface: http://0.0.0.0:%d", WEB_PORT)
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, debug=False)