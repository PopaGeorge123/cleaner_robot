"""
lidar_ld06.py
-------------
Pure Python serial driver for the LDRobot LD06 360° laser range scanner.

No ROS required.  Dependencies: pyserial, numpy.

The LD06 outputs packets over a serial UART at 230400 baud.
Each packet contains 12 measurement points.

Packet structure (47 bytes total)
----------------------------------
Byte  0      : 0x54  (header)
Byte  1      : 0x2C  (verlen – always 0x2C for LD06)
Bytes 2–3    : speed (degrees/second, little-endian uint16)
Bytes 4–5    : start_angle (0.01°, little-endian uint16)
Bytes 6–41   : 12 × (distance uint16 LE + intensity uint8)  = 36 bytes
Bytes 42–43  : end_angle (0.01°, little-endian uint16)
Bytes 44–45  : timestamp (ms, little-endian uint16)
Byte  46     : CRC-8 checksum

Usage – live scan
-----------------
    from lidar_ld06 import LD06Driver

    lidar = LD06Driver(port="/dev/ttyUSB0")
    lidar.start()

    while True:
        scan = lidar.get_scan()   # blocks until a full 360° scan is ready
        if scan:
            for angle_deg, dist_m, intensity in scan:
                print(f"  {angle_deg:.1f}°  {dist_m:.3f} m  int={intensity}")

    lidar.stop()

Usage – non-blocking callback
------------------------------
    def on_scan(scan):
        nearest = min(scan, key=lambda p: p[1])
        print(f"Nearest obstacle: {nearest[1]:.3f} m at {nearest[0]:.1f}°")

    lidar = LD06Driver(port="/dev/ttyUSB0", scan_callback=on_scan)
    lidar.start()
    ...
    lidar.stop()

Simulated / offline usage
--------------------------
    from lidar_ld06 import SimulatedLD06, polar_to_cartesian

    sim = SimulatedLD06(obstacles=[(2.0, 0.0), (1.0, 1.0)])
    scan = sim.get_scan()
"""

from __future__ import annotations
import math
import struct
import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np

from robot_config import (
    LIDAR_SERIAL_PORT, LIDAR_BAUD_RATE,
    LIDAR_RANGE_MIN_M, LIDAR_RANGE_MAX_M,
    LIDAR_SAMPLES, LIDAR_SCAN_FREQ_HZ,
    LIDAR_PACKET_HEADER, LIDAR_POINTS_PER_PKT,
    LIDAR_MOUNT_X, LIDAR_MOUNT_Y, LIDAR_MOUNT_Z,
)


# Type alias:  list of (angle_deg, distance_m, intensity)
ScanPoint  = Tuple[float, float, int]
Scan       = List[ScanPoint]

# ── CRC-8 table (polynomial 0x4D) ─────────────────────────────────────────────
_CRC_TABLE = [0] * 256
for _i in range(256):
    _crc = _i
    for _ in range(8):
        if _crc & 0x80:
            _crc = ((_crc << 1) ^ 0x4D) & 0xFF
        else:
            _crc = (_crc << 1) & 0xFF
    _CRC_TABLE[_i] = _crc


def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


# ── Live driver ────────────────────────────────────────────────────────────────

class LD06Driver:
    """
    Serial driver for the LD06 lidar.

    Parameters
    ----------
    port          : str   serial device, e.g. '/dev/ttyUSB0'
    baud_rate     : int   default 230400
    scan_callback : callable(Scan) or None
                    called in the reader thread each time a full 360° scan
                    is assembled.  If None use get_scan() instead.
    """

    PACKET_SIZE = 47
    HEADER      = LIDAR_PACKET_HEADER
    VERLEN      = 0x2C

    def __init__(
        self,
        port:          str      = LIDAR_SERIAL_PORT,
        baud_rate:     int      = LIDAR_BAUD_RATE,
        scan_callback: Optional[Callable[[Scan], None]] = None,
    ):
        self.port          = port
        self.baud_rate     = baud_rate
        self.scan_callback = scan_callback

        self._serial    = None
        self._thread    = None
        self._running   = False

        # Accumulate points until we've covered ≥ 360°
        self._scan_buf:  Scan  = []
        self._last_angle: float = -1.0
        self._scan_queue: deque = deque(maxlen=5)

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open serial port and start the reader thread."""
        try:
            import serial
        except ImportError:
            raise ImportError(
                "pyserial is not installed.  Run:  pip install pyserial"
            )

        self._serial  = serial.Serial(self.port, self.baud_rate, timeout=1.0)
        self._running = True
        self._thread  = threading.Thread(target=self._reader_loop, daemon=True,
                                         name="LD06-reader")
        self._thread.start()
        print(f"[LD06] Started on {self.port} at {self.baud_rate} baud.")

    def stop(self) -> None:
        """Stop the reader thread and close the serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        print("[LD06] Stopped.")

    # ── Data access ───────────────────────────────────────────────────────────

    def get_scan(self, timeout: float = 2.0) -> Optional[Scan]:
        """
        Block until a complete 360° scan is available and return it.

        Returns None on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._scan_queue:
                return self._scan_queue.popleft()
            time.sleep(0.005)
        return None

    def latest_scan(self) -> Optional[Scan]:
        """Return the most recent scan without blocking (None if none yet)."""
        if self._scan_queue:
            return self._scan_queue[-1]
        return None

    # ── Reader thread ─────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        buf = bytearray()
        while self._running:
            try:
                chunk = self._serial.read(self._serial.in_waiting or 1)
                if chunk:
                    buf.extend(chunk)
                    buf = self._process_buffer(buf)
            except Exception as e:
                print(f"[LD06] Serial error: {e}")
                time.sleep(0.1)

    def _process_buffer(self, buf: bytearray) -> bytearray:
        """Parse as many complete packets from buf as possible."""
        while len(buf) >= self.PACKET_SIZE:
            # Find header byte
            if buf[0] != self.HEADER or buf[1] != self.VERLEN:
                buf.pop(0)
                continue

            pkt = bytes(buf[:self.PACKET_SIZE])
            buf = buf[self.PACKET_SIZE:]

            # CRC check (covers first 46 bytes)
            if _crc8(pkt[:46]) != pkt[46]:
                continue  # discard corrupt packet

            self._parse_packet(pkt)

        return buf

    def _parse_packet(self, pkt: bytes) -> None:
        """Decode one 47-byte packet and add points to scan buffer."""
        # speed = struct.unpack_from('<H', pkt, 2)[0]   # deg/s (unused)
        start_angle = struct.unpack_from('<H', pkt, 4)[0] / 100.0   # degrees
        end_angle   = struct.unpack_from('<H', pkt, 42)[0] / 100.0  # degrees
        # timestamp = struct.unpack_from('<H', pkt, 44)[0]           # ms

        n = LIDAR_POINTS_PER_PKT
        # Angular step between points in this packet
        if end_angle > start_angle:
            step = (end_angle - start_angle) / (n - 1)
        else:
            # Wrap-around (e.g. 350° → 10°)
            step = (end_angle + 360.0 - start_angle) / (n - 1)

        for i in range(n):
            offset    = 6 + i * 3
            dist_mm   = struct.unpack_from('<H', pkt, offset)[0]
            intensity = pkt[offset + 2]

            angle_deg = (start_angle + i * step) % 360.0
            dist_m    = dist_mm / 1000.0

            # Filter invalid ranges
            if dist_m < LIDAR_RANGE_MIN_M or dist_m > LIDAR_RANGE_MAX_M:
                continue

            # Detect scan wrap (new revolution started)
            if self._last_angle > 300.0 and angle_deg < 60.0:
                if len(self._scan_buf) > LIDAR_SAMPLES // 2:
                    scan = list(self._scan_buf)
                    self._scan_queue.append(scan)
                    if self.scan_callback:
                        self.scan_callback(scan)
                self._scan_buf = []

            self._scan_buf.append((angle_deg, dist_m, intensity))
            self._last_angle = angle_deg


# ── Utilities ─────────────────────────────────────────────────────────────────

def polar_to_cartesian(
    scan: Scan,
    mount_x: float = LIDAR_MOUNT_X,
    mount_y: float = LIDAR_MOUNT_Y,
) -> List[Tuple[float, float]]:
    """
    Convert a polar scan to (x, y) Cartesian points in the robot frame,
    accounting for the lidar's mount offset.

    Parameters
    ----------
    scan    : Scan  – list of (angle_deg, dist_m, intensity)
    mount_x : float – forward offset of lidar from base_link  (m)
    mount_y : float – lateral offset of lidar from base_link  (m)

    Returns
    -------
    List of (x, y) in metres relative to base_link.
    """
    points = []
    for angle_deg, dist_m, _ in scan:
        a = math.radians(angle_deg)
        lx = dist_m * math.cos(a)
        ly = dist_m * math.sin(a)
        points.append((lx + mount_x, ly + mount_y))
    return points


def nearest_obstacle(scan: Scan) -> Optional[ScanPoint]:
    """Return the (angle, dist, intensity) of the closest valid reading."""
    if not scan:
        return None
    return min(scan, key=lambda p: p[1])


def scan_to_numpy(scan: Scan) -> np.ndarray:
    """
    Convert a Scan to a NumPy array of shape (N, 3):
    columns = [angle_deg, dist_m, intensity]
    """
    return np.array(scan, dtype=np.float32)


def scan_sector(
    scan: Scan,
    angle_min: float,
    angle_max: float,
) -> Scan:
    """
    Return only the points whose angle is within [angle_min, angle_max] degrees.
    Both bounds are in [0, 360).
    """
    if angle_min <= angle_max:
        return [(a, d, i) for a, d, i in scan if angle_min <= a <= angle_max]
    else:
        # Wrap-around sector, e.g. 350° → 10°
        return [(a, d, i) for a, d, i in scan if a >= angle_min or a <= angle_max]


# ── Simulated LD06 (for testing without hardware) ─────────────────────────────

class SimulatedLD06:
    """
    Simulate an LD06 scan given a list of obstacle (x, y) world positions
    and the robot's current pose.  Useful for testing without hardware.

    Parameters
    ----------
    obstacles : list of (x, y) in robot frame (metres)
    noise_m   : float – Gaussian range noise standard deviation
    """

    def __init__(
        self,
        obstacles: List[Tuple[float, float]] | None = None,
        noise_m:   float = 0.01,
    ):
        self.obstacles = obstacles or []
        self.noise_m   = noise_m

    def get_scan(self) -> Scan:
        """Generate a synthetic 360° scan."""
        scan: Scan = []
        for deg in range(LIDAR_SAMPLES):
            angle_rad = math.radians(deg)
            ax = math.cos(angle_rad)
            ay = math.sin(angle_rad)

            best_dist = LIDAR_RANGE_MAX_M
            for ox, oy in self.obstacles:
                # Ray-point distance (project obstacle onto beam ray)
                t = ox * ax + oy * ay
                if t < LIDAR_RANGE_MIN_M:
                    continue
                perp = math.hypot(ox - t * ax, oy - t * ay)
                if perp < 0.1:   # 10 cm hit radius
                    best_dist = min(best_dist, t)

            if best_dist < LIDAR_RANGE_MAX_M:
                noise = np.random.normal(0, self.noise_m)
                dist  = max(LIDAR_RANGE_MIN_M, best_dist + noise)
                scan.append((float(deg), dist, 200))
            else:
                # No obstacle in this direction – report max range (filtered out)
                pass

        return scan


# ── Quick test / demo ──────────────────────────────────────────────────────────

def demo_simulated() -> None:
    """Print a simulated scan and visualise it with matplotlib."""
    obstacles = [(1.5, 0.0), (0.0, 2.0), (-1.0, 1.0), (2.0, -1.5)]
    sim  = SimulatedLD06(obstacles=obstacles)
    scan = sim.get_scan()
    print(f"[LD06-sim] Generated {len(scan)} scan points.")

    nearest = nearest_obstacle(scan)
    if nearest:
        print(f"[LD06-sim] Nearest obstacle: {nearest[1]:.3f} m at {nearest[0]:.1f}°")

    # Try to plot
    try:
        import matplotlib.pyplot as plt
        pts = polar_to_cartesian(scan, mount_x=0, mount_y=0)
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        angles  = [math.radians(p[0]) for p in scan]
        dists   = [p[1] for p in scan]
        ax.scatter(angles, dists, s=5, c='cyan')
        for ox, oy in obstacles:
            a = math.atan2(oy, ox)
            d = math.hypot(ox, oy)
            ax.plot(a, d, 'r*', markersize=10)
        ax.set_title("LD06 Simulated Scan (red = obstacles)")
        ax.set_ylim(0, LIDAR_RANGE_MAX_M)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[LD06-sim] matplotlib not installed – skipping plot.")


if __name__ == '__main__':
    demo_simulated()
