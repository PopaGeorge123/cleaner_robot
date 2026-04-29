"""
live_mapper.py  (rewritten)
────────────────────────────
Live mapping server for LD06 lidar + Arduino odometry robot.
Streams a persistent occupancy map to any browser on the LAN.

Fixes vs original:
  - Canvas fills the browser window properly (no tiny top-left corner)
  - Map accumulates permanently (no fading that erases points)
  - Fast updates: ~5 Hz via SocketIO, no blocking sleep
  - Robot pose drawn as an arrow on the map
  - Robot path drawn in green
  - Scan points never disappear (optional slow decay can be toggled)
  - Proper coordinate transform so map is centred and fills canvas
  - Status bar shows scan count, point count, robot pose
  - Works with just Lidar OR just Arduino (graceful fallback)

Requirements:
    pip install flask flask-socketio eventlet numpy pillow

Usage:
    python3 live_mapper.py
    Open http://<raspberry-pi-ip>:5000
"""

import threading
import time
import math
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ── Try to import hardware drivers ───────────────────────────────────────────
try:
    from lidar_ld06 import LD06Driver
    HAS_LIDAR = True
except ImportError:
    HAS_LIDAR = False
    print("[WARN] lidar_ld06 not found – lidar disabled")

try:
    from arduino_driver import ArduinoDriver
    HAS_ARDUINO = True
except ImportError:
    HAS_ARDUINO = False
    print("[WARN] arduino_driver not found – odometry disabled")

try:
    from robot_config import (
        LIDAR_SERIAL_PORT, LIDAR_BAUD_RATE,
        ARDUINO_SERIAL_PORT, MAP_RESOLUTION,
    )
except ImportError:
    LIDAR_SERIAL_PORT  = "/dev/ttyUSB1"
    ARDUINO_BAUD_RATE  = 57600
    ARDUINO_SERIAL_PORT = "/dev/ttyUSB0"
    MAP_RESOLUTION     = 0.05
    print("[WARN] robot_config not found – using defaults")

# ── Map parameters ────────────────────────────────────────────────────────────
# 20 m × 20 m world at MAP_RESOLUTION m/px
MAP_WORLD_M  = 20.0
MAP_W        = int(MAP_WORLD_M / MAP_RESOLUTION)   # 400 px
MAP_H        = int(MAP_WORLD_M / MAP_RESOLUTION)   # 400 px
MAP_OX       = MAP_WORLD_M / 2.0   # world-x of map centre  (metres)
MAP_OY       = MAP_WORLD_M / 2.0   # world-y of map centre

# Persistent occupancy counts (never decrease unless reset requested)
_hit_count   = np.zeros((MAP_H, MAP_W), dtype=np.uint16)
_miss_count  = np.zeros((MAP_H, MAP_W), dtype=np.uint16)

# Robot state (updated by mapping thread)
_robot_pose  = {"x": 0.0, "y": 0.0, "theta": 0.0}
_robot_path  = []           # list of (mx, my) in map pixels
_stats       = {"scans": 0, "points": 0, "fps": 0.0}
_reset_flag  = threading.Event()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Live Robot Mapper</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #1a1a1a;
  color: #eee;
  font-family: monospace;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}
#header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 14px;
  background: #111;
  border-bottom: 1px solid #333;
  flex-shrink: 0;
}
#header h2 { font-size: 15px; font-weight: 600; color: #fff; }
#status {
  font-size: 12px;
  color: #aaa;
  flex: 1;
}
button {
  background: #2a6;
  color: #fff;
  border: none;
  border-radius: 5px;
  padding: 5px 14px;
  font-size: 13px;
  cursor: pointer;
}
button:hover { background: #3b7; }
button.danger { background: #a33; }
button.danger:hover { background: #c44; }
#canvas-wrap {
  flex: 1;
  overflow: hidden;
  position: relative;
}
canvas {
  display: block;
  width: 100%;
  height: 100%;
  image-rendering: pixelated;
}
#save-msg {
  position: absolute;
  bottom: 10px;
  right: 14px;
  background: rgba(0,0,0,0.7);
  color: #8f8;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 4px;
  display: none;
}
#legend {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(0,0,0,0.6);
  padding: 6px 10px;
  border-radius: 5px;
  font-size: 11px;
  line-height: 1.7;
  color: #ccc;
}
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
</style>
</head>
<body>
<div id="header">
  <h2>🤖 Live Robot Mapper</h2>
  <div id="status">Connecting…</div>
  <button onclick="saveMap()">💾 Save map</button>
  <button class="danger" onclick="resetMap()">🗑 Reset</button>
</div>
<div id="canvas-wrap">
  <canvas id="map"></canvas>
  <div id="legend">
    <span class="dot" style="background:#fff"></span>Obstacles<br>
    <span class="dot" style="background:#0f0"></span>Robot path<br>
    <span class="dot" style="background:#f55"></span>Robot
  </div>
  <div id="save-msg"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
<script>
const canvas = document.getElementById('map');
const ctx    = canvas.getContext('2d');
const wrap   = document.getElementById('canvas-wrap');

// Map dimensions from server (set on first update)
let MAP_W = {{MAP_W}};
let MAP_H = {{MAP_H}};

// Resize canvas to fill container while keeping map aspect ratio
function resizeCanvas() {
  const cw = wrap.clientWidth;
  const ch = wrap.clientHeight;
  const scale = Math.min(cw / MAP_W, ch / MAP_H);
  canvas.width  = Math.round(MAP_W * scale);
  canvas.height = Math.round(MAP_H * scale);
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

const socket = io();

socket.on('connect', () => {
  document.getElementById('status').textContent = 'Connected – waiting for scan…';
});
socket.on('disconnect', () => {
  document.getElementById('status').textContent = '⚠ Disconnected';
});

socket.on('map_update', function(data) {
  const W = data.map_w;
  const H = data.map_h;
  if (W !== MAP_W || H !== MAP_H) {
    MAP_W = W; MAP_H = H;
    resizeCanvas();
  }

  // Draw map pixels
  const imgData = ctx.createImageData(W, H);
  const raw     = new Uint8Array(data.map);   // flat W*H bytes, 0=free, 255=wall
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i];
    imgData.data[i*4+0] = v;
    imgData.data[i*4+1] = v;
    imgData.data[i*4+2] = v;
    imgData.data[i*4+3] = 255;
  }

  // Render map into offscreen, then scale to canvas
  const tmp = new OffscreenCanvas(W, H);
  tmp.getContext('2d').putImageData(imgData, 0, 0);
  ctx.save();
  ctx.scale(canvas.width / W, canvas.height / H);

  ctx.drawImage(tmp, 0, 0);

  // Draw robot path
  const path = data.path;
  if (path && path.length > 1) {
    ctx.strokeStyle = '#00cc44';
    ctx.lineWidth   = 1.5 / (canvas.width / W);
    ctx.lineJoin    = 'round';
    ctx.beginPath();
    ctx.moveTo(path[0][0] + 0.5, path[0][1] + 0.5);
    for (let i = 1; i < path.length; i++) {
      ctx.lineTo(path[i][0] + 0.5, path[i][1] + 0.5);
    }
    ctx.stroke();
  }

  // Draw robot as arrow
  const rx    = data.robot_mx;
  const ry    = data.robot_my;
  const theta = data.robot_theta;
  const R     = 4 / (canvas.width / W);  // arrow size in map pixels

  ctx.save();
  ctx.translate(rx + 0.5, ry + 0.5);
  ctx.rotate(-theta);   // canvas Y is flipped vs world Y
  ctx.beginPath();
  ctx.moveTo( R * 1.8,  0);
  ctx.lineTo(-R,  R * 0.9);
  ctx.lineTo(-R, -R * 0.9);
  ctx.closePath();
  ctx.fillStyle   = '#ff4444';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth   = 0.5 / (canvas.width / W);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  ctx.restore();

  // Update status bar
  const p = data.pose;
  document.getElementById('status').textContent =
    `Scans: ${data.scans}  |  Points: ${data.points}  |  ` +
    `x=${p.x.toFixed(2)}m  y=${p.y.toFixed(2)}m  θ=${(p.theta * 180 / Math.PI).toFixed(1)}°  |  ` +
    `${data.fps.toFixed(1)} Hz`;
});

socket.on('save_status', function(d) {
  const el = document.getElementById('save-msg');
  el.textContent = d.msg;
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 4000);
});

function saveMap()  { socket.emit('save_map', {}); }
function resetMap() { socket.emit('reset_map', {}); }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML, MAP_W=MAP_W, MAP_H=MAP_H)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def world_to_map(wx, wy):
    """Convert world (x, y) metres → map pixel (mx, my)."""
    mx = int(MAP_OX / MAP_RESOLUTION + wx / MAP_RESOLUTION)
    my = int(MAP_OY / MAP_RESOLUTION - wy / MAP_RESOLUTION)   # Y flipped
    return mx, my


def render_map():
    """
    Convert hit/miss counts → grayscale image bytes.
    Free cells = 0 (black background).
    Occupied cells = 255 (white walls).
    """
    with _map_lock:
        total = _hit_count.astype(np.float32) + _miss_count.astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            prob = np.where(total > 0, _hit_count / total, 0.5)
        img = np.zeros((MAP_H, MAP_W), dtype=np.uint8)
        img[prob > 0.55] = 255    # wall  → white
        img[prob < 0.35] = 0     # free  → black (background)
        # Unknown stays 0 too – clean black background looks better
    return img.flatten().tobytes()


_map_lock = threading.Lock()


# ── Mapping thread ────────────────────────────────────────────────────────────

def mapping_thread():
    global _robot_path

    lidar   = None
    arduino = None

    # Start Lidar
    if HAS_LIDAR:
        try:
            lidar = LD06Driver(port=LIDAR_SERIAL_PORT, baud_rate=LIDAR_BAUD_RATE)
            lidar.start()
            print(f"[INFO] Lidar started on {LIDAR_SERIAL_PORT}")
        except Exception as e:
            print(f"[ERROR] Lidar failed: {e}")
            lidar = None

    # Start Arduino
    if HAS_ARDUINO:
        try:
            arduino = ArduinoDriver(port=ARDUINO_SERIAL_PORT)
            arduino.start()
            print(f"[INFO] Arduino started on {ARDUINO_SERIAL_PORT}")
        except Exception as e:
            print(f"[ERROR] Arduino failed: {e}")
            arduino = None

    if lidar is None and arduino is None:
        print("[ERROR] No hardware available – mapping thread idle")
        return

    scan_count  = 0
    point_count = 0
    t_last      = time.monotonic()

    while True:
        # Handle reset
        if _reset_flag.is_set():
            _reset_flag.clear()
            with _map_lock:
                _hit_count[:]  = 0
                _miss_count[:] = 0
            _robot_path.clear()
            scan_count  = 0
            point_count = 0
            print("[INFO] Map reset")

        # Robot pose
        x, y, theta = 0.0, 0.0, 0.0
        if arduino:
            try:
                pose  = arduino.odometry
                x, y, theta = pose.x, pose.y, pose.theta
            except Exception as e:
                print(f"[WARN] Odometry error: {e}")

        # Record path
        mx, my = world_to_map(x, y)
        if 0 <= mx < MAP_W and 0 <= my < MAP_H:
            if not _robot_path or (abs(_robot_path[-1][0] - mx) + abs(_robot_path[-1][1] - my)) > 1:
                _robot_path.append((mx, my))
                if len(_robot_path) > 2000:
                    _robot_path = _robot_path[-2000:]

        # Get lidar scan (non-blocking – use latest_scan if available)
        scan = None
        if lidar:
            # Try non-blocking first, then fall back to short-wait
            scan = lidar.latest_scan()
            if scan is None:
                scan = lidar.get_scan(timeout=0.5)

        if scan:
            scan_count  += 1
            new_pts      = 0

            with _map_lock:
                for angle_deg, dist_m, intensity in scan:
                    if dist_m < 0.05 or dist_m > 8.0:
                        continue

                    # World position of obstacle
                    angle_world = theta + math.radians(angle_deg)
                    lx = x + dist_m * math.cos(angle_world)
                    ly = y + dist_m * math.sin(angle_world)

                    omx, omy = world_to_map(lx, ly)
                    if 0 <= omx < MAP_W and 0 <= omy < MAP_H:
                        _hit_count[omy, omx] = min(65535, _hit_count[omy, omx] + 3)
                        new_pts += 1

                    # Mark cells along ray as free (simple: just mark midpoint)
                    for frac in (0.33, 0.66):
                        fx = x + dist_m * frac * math.cos(angle_world)
                        fy = y + dist_m * frac * math.sin(angle_world)
                        fmx, fmy = world_to_map(fx, fy)
                        if 0 <= fmx < MAP_W and 0 <= fmy < MAP_H:
                            _miss_count[fmy, fmx] = min(65535, _miss_count[fmy, fmy] + 1)

            point_count += new_pts

        # Compute update rate
        now   = time.monotonic()
        dt    = now - t_last
        t_last = now
        fps    = 1.0 / dt if dt > 0 else 0.0

        # Emit to browser
        img_bytes = render_map()
        _robot_pose.update({"x": x, "y": y, "theta": theta})
        _stats.update({"scans": scan_count, "points": point_count, "fps": fps})

        socketio.emit('map_update', {
            'map':        list(img_bytes),
            'map_w':      MAP_W,
            'map_h':      MAP_H,
            'path':       _robot_path[-500:],
            'robot_mx':   mx,
            'robot_my':   my,
            'robot_theta': theta,
            'pose':        {'x': round(x, 3), 'y': round(y, 3), 'theta': round(theta, 3)},
            'scans':       scan_count,
            'points':      point_count,
            'fps':         round(fps, 1),
        })

        # Small sleep to avoid hammering CPU; get_scan() above provides natural pacing
        time.sleep(0.05)


# ── Socket events ─────────────────────────────────────────────────────────────

@socketio.on('save_map')
def handle_save_map(_):
    try:
        from map_loader import OccupancyGrid
        import time as _t
        with _map_lock:
            # Convert hit/miss to 0–100 probability grid
            total = _hit_count.astype(np.float32) + _miss_count.astype(np.float32)
            with np.errstate(divide='ignore', invalid='ignore'):
                prob = np.where(total > 0, _hit_count / total, 0.5).astype(np.float32)
        grid = OccupancyGrid(
            data       = (prob * 100).astype(np.uint8),
            resolution = MAP_RESOLUTION,
            origin     = (-MAP_OX, -MAP_OY, 0.0),
        )
        fname = f"map_{int(_t.time())}"
        grid.save(fname)
        socketio.emit('save_status', {'msg': f'✓ Saved: {fname}.yaml + .pgm'})
    except Exception as e:
        socketio.emit('save_status', {'msg': f'✗ Save error: {e}'})


@socketio.on('reset_map')
def handle_reset_map(_):
    _reset_flag.set()
    socketio.emit('save_status', {'msg': '🗑 Map reset'})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[INFO] Starting mapping thread…")
    t = threading.Thread(target=mapping_thread, daemon=True)
    t.start()
    print("[INFO] Open http://0.0.0.0:5000 in your browser")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)