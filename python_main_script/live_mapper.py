"""
live_mapper.py  (v3 - truly live)
──────────────────────────────────
Key fixes vs previous version:
  - Lidar runs in its OWN thread via callback → no blocking get_scan()
  - Map updates sent at 10 Hz regardless of scan rate
  - Delta compression: only changed pixels sent over the wire
  - Canvas fills browser, top-down view, robot arrow, green path
  - Save as PGM + YAML (ROS-compatible) + PNG

Requirements:
    pip install flask flask-socketio eventlet numpy pillow pyserial

Run:
    python3 live_mapper.py
    Open http://<raspberry-pi-ip>:5000
"""

import threading
import time
import math
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ── Config ────────────────────────────────────────────────────────────────────
LIDAR_PORT   = "/dev/ttyUSB1"
LIDAR_BAUD   = 230400
ARDUINO_PORT = "/dev/ttyUSB0"
ARDUINO_BAUD = 57600
MAP_RESOLUTION = 0.05   # metres per pixel
MAP_WORLD_M    = 20.0   # map covers 20m x 20m
SEND_HZ        = 10     # browser update rate

try:
    from robot_config import (
        LIDAR_SERIAL_PORT, LIDAR_BAUD_RATE,
        ARDUINO_SERIAL_PORT, ARDUINO_BAUD_RATE,
        MAP_RESOLUTION,
    )
    LIDAR_PORT   = LIDAR_SERIAL_PORT
    LIDAR_BAUD   = LIDAR_BAUD_RATE
    ARDUINO_PORT = ARDUINO_SERIAL_PORT
    ARDUINO_BAUD = ARDUINO_BAUD_RATE
except Exception:
    pass

MAP_W  = int(MAP_WORLD_M / MAP_RESOLUTION)   # 400 px
MAP_H  = int(MAP_WORLD_M / MAP_RESOLUTION)
MAP_CX = MAP_W // 2
MAP_CY = MAP_H // 2

# ── Shared state ──────────────────────────────────────────────────────────────
_lock      = threading.Lock()
_map       = np.zeros((MAP_H, MAP_W), dtype=np.uint8)   # 0=unseen, 1-254=free, 255=wall
_prev_map  = np.zeros((MAP_H, MAP_W), dtype=np.uint8)
_path_px   = []
_pose      = [0.0, 0.0, 0.0]   # x, y, theta
_stats     = {"scans": 0, "pts": 0, "lidar_ok": False, "arduino_ok": False}
_reset_evt = threading.Event()

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet",
                    max_http_buffer_size=2_000_000)


# ── Coordinate helper ─────────────────────────────────────────────────────────
def w2m(wx, wy):
    mx = int(MAP_CX + wx / MAP_RESOLUTION)
    my = int(MAP_CY - wy / MAP_RESOLUTION)
    return mx, my


# ── Bresenham line ────────────────────────────────────────────────────────────
def _bresenham(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cx, cy = x0, y0
    for _ in range(1200):
        pts.append((cx, cy))
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 > -dy: err -= dy; cx += sx
        if e2 <  dx: err += dx; cy += sy
    return pts


# ── Lidar scan callback (runs inside LD06 reader thread) ─────────────────────
def on_scan(scan):
    with _lock:
        x, y, theta = _pose[0], _pose[1], _pose[2]

    rmx, rmy = w2m(x, y)

    with _lock:
        for angle_deg, dist_m, intensity in scan:
            if dist_m < 0.05 or dist_m > 10.0:
                continue
            a  = theta + math.radians(angle_deg)
            ox = x + dist_m * math.cos(a)
            oy = y + dist_m * math.sin(a)
            omx, omy = w2m(ox, oy)

            # Mark wall
            if 0 <= omx < MAP_W and 0 <= omy < MAP_H:
                _map[omy, omx] = 255

            # Bresenham ray: mark free cells
            ray = _bresenham(rmx, rmy, omx, omy)
            for bx, by in ray[:-1]:
                if 0 <= bx < MAP_W and 0 <= by < MAP_H:
                    if _map[by, bx] < 255:
                        _map[by, bx] = max(_map[by, bx], 80)

        _stats["scans"] += 1
        _stats["pts"]   += len(scan)
        _stats["lidar_ok"] = True


# ── Arduino odometry thread ───────────────────────────────────────────────────
def arduino_thread():
    METRES_PER_TICK  = (2 * math.pi * 0.033) / 3436
    WHEEL_SEPARATION = 0.297

    try:
        import serial
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.5)
        time.sleep(2.0)
        ser.reset_input_buffer()
        with _lock:
            _stats["arduino_ok"] = True
        print(f"[Arduino] Connected on {ARDUINO_PORT}")
    except Exception as e:
        print(f"[Arduino] FAILED: {e}")
        return

    def send(cmd):
        try:
            ser.reset_input_buffer()
            ser.write((cmd + "\r").encode())
            return ser.readline().decode(errors="replace")
        except Exception:
            return ""

    prev_l = prev_r = 0
    while True:
        try:
            raw   = send("e")
            parts = raw.strip().split()
            if len(parts) >= 2:
                nl, nr = int(parts[0]), int(parts[1])
                dl, dr = nl - prev_l, nr - prev_r
                prev_l, prev_r = nl, nr

                dist_l = dl * METRES_PER_TICK
                dist_r = dr * METRES_PER_TICK
                dist   = (dist_l + dist_r) / 2.0
                dtheta = (dist_r - dist_l) / WHEEL_SEPARATION

                with _lock:
                    _pose[2] += dtheta
                    _pose[0] += dist * math.cos(_pose[2] - dtheta / 2)
                    _pose[1] += dist * math.sin(_pose[2] - dtheta / 2)
                    mx, my = w2m(_pose[0], _pose[1])
                    if 0 <= mx < MAP_W and 0 <= my < MAP_H:
                        if not _path_px or _path_px[-1] != (mx, my):
                            _path_px.append((mx, my))
                            if len(_path_px) > 5000:
                                del _path_px[:1000]
        except Exception:
            pass
        time.sleep(0.033)


# ── Map sender thread ─────────────────────────────────────────────────────────
def sender_thread():
    interval = 1.0 / SEND_HZ

    while True:
        t0 = time.monotonic()

        if _reset_evt.is_set():
            _reset_evt.clear()
            with _lock:
                _map[:] = 0
                _prev_map[:] = 0
                _path_px.clear()
                _pose[0] = _pose[1] = _pose[2] = 0.0
                _stats["scans"] = _stats["pts"] = 0
            print("[INFO] Map reset")

        with _lock:
            snap      = _map.copy()
            path_snap = list(_path_px[-500:])
            px, py, ptheta = _pose[0], _pose[1], _pose[2]
            sc = dict(_stats)

        # Delta: only send pixels that changed
        changed = np.where(snap != _prev_map)
        rows = changed[0].tolist()
        cols = changed[1].tolist()
        vals = snap[changed].tolist()
        _prev_map[:] = snap

        mx, my = w2m(px, py)

        socketio.emit('map_update', {
            'rows':    rows,
            'cols':    cols,
            'vals':    vals,
            'map_w':   MAP_W,
            'map_h':   MAP_H,
            'path':    path_snap,
            'rmx':     mx,
            'rmy':     my,
            'rtheta':  ptheta,
            'pose':    {'x': round(px,3), 'y': round(py,3), 'theta': round(ptheta,3)},
            'scans':   sc['scans'],
            'pts':     sc['pts'],
            'lidar':   sc['lidar_ok'],
            'arduino': sc['arduino_ok'],
        })

        elapsed = time.monotonic() - t0
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ── Socket events ─────────────────────────────────────────────────────────────
@socketio.on('request_full')
def handle_full(_):
    """Send entire map on (re)connect."""
    with _lock:
        snap      = _map.copy()
        path_snap = list(_path_px[-500:])
        px, py, ptheta = _pose[0], _pose[1], _pose[2]
        sc = dict(_stats)
    rows, cols = np.where(snap > 0)
    mx, my = w2m(px, py)
    socketio.emit('map_update', {
        'rows':    rows.tolist(),
        'cols':    cols.tolist(),
        'vals':    snap[rows, cols].tolist(),
        'map_w':   MAP_W,
        'map_h':   MAP_H,
        'path':    path_snap,
        'rmx':     mx, 'rmy': my, 'rtheta': ptheta,
        'pose':    {'x': round(px,3), 'y': round(py,3), 'theta': round(ptheta,3)},
        'scans':   sc['scans'], 'pts': sc['pts'],
        'lidar':   sc['lidar_ok'], 'arduino': sc['arduino_ok'],
    })


@socketio.on('reset_map')
def handle_reset(_):
    _reset_evt.set()


@socketio.on('save_map')
def handle_save(_):
    try:
        with _lock:
            snap = _map.copy()
        ts    = int(time.time())
        fname = f"map_{ts}"

        # PGM (ROS: black=occupied, white=free, grey=unknown)
        pgm = np.full((MAP_H, MAP_W), 205, dtype=np.uint8)
        pgm[snap == 255] = 0
        pgm[(snap > 0) & (snap < 255)] = 254

        with open(fname + ".pgm", "wb") as f:
            f.write(f"P5\n{MAP_W} {MAP_H}\n255\n".encode())
            f.write(pgm.tobytes())

        ox = -MAP_CX * MAP_RESOLUTION
        oy = -MAP_CY * MAP_RESOLUTION
        with open(fname + ".yaml", "w") as f:
            f.write(
                f"image: {fname}.pgm\n"
                f"resolution: {MAP_RESOLUTION}\n"
                f"origin: [{ox:.4f}, {oy:.4f}, 0.0]\n"
                f"negate: 0\n"
                f"occupied_thresh: 0.65\n"
                f"free_thresh: 0.196\n"
            )

        try:
            from PIL import Image as PILImage
            PILImage.fromarray(pgm).save(fname + ".png")
            msg = f"✓ Saved {fname}.pgm / .yaml / .png"
        except ImportError:
            msg = f"✓ Saved {fname}.pgm / .yaml  (install pillow for .png)"

        print(f"[INFO] {msg}")
        socketio.emit('save_status', {'ok': True,  'msg': msg})
    except Exception as e:
        socketio.emit('save_status', {'ok': False, 'msg': f'✗ Error: {e}'})


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Live Robot Mapper</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#ddd;font-family:monospace;display:flex;flex-direction:column;height:100vh;overflow:hidden}
#hdr{display:flex;align-items:center;gap:10px;padding:6px 12px;background:#1a1a1a;border-bottom:1px solid #333;flex-shrink:0}
#hdr h2{font-size:14px;color:#fff;white-space:nowrap}
#status{font-size:11px;color:#aaa;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{padding:2px 8px;border-radius:10px;font-size:11px;background:#333;color:#666;white-space:nowrap}
.pill.ok{background:#1a3a1a;color:#4c4}
.pill.err{background:#3a1a1a;color:#c44}
button{background:#2a6;color:#fff;border:none;border-radius:4px;padding:4px 12px;font-size:12px;cursor:pointer;white-space:nowrap}
button:hover{background:#3b7}
button.warn{background:#a33}
button.warn:hover{background:#c44}
#wrap{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative;background:#0a0a0a}
canvas{image-rendering:pixelated;image-rendering:crisp-edges;display:block}
#save-msg{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.88);font-size:13px;padding:7px 20px;border-radius:6px;display:none;white-space:nowrap;pointer-events:none}
#legend{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.72);padding:7px 11px;border-radius:5px;font-size:11px;line-height:2;color:#bbb}
.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
</style>
</head>
<body>
<div id="hdr">
  <h2>🤖 Live Robot Mapper</h2>
  <span id="pill-lidar" class="pill">Lidar ✗</span>
  <span id="pill-ard"   class="pill">Arduino ✗</span>
  <div id="status">Connecting…</div>
  <button onclick="saveMap()">💾 Save map</button>
  <button class="warn" onclick="resetMap()">🗑 Reset</button>
</div>
<div id="wrap">
  <canvas id="map"></canvas>
  <div id="legend">
    <span class="dot" style="background:#fff"></span>Wall<br>
    <span class="dot" style="background:#333;border:1px solid #555"></span>Free space<br>
    <span class="dot" style="background:#0a0a0a;border:1px solid #444"></span>Unseen<br>
    <span class="dot" style="background:#0c0"></span>Path<br>
    <span class="dot" style="background:#f44;border-radius:50%"></span>Robot
  </div>
  <div id="save-msg"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
<script>
const wrap = document.getElementById('wrap');
const cv   = document.getElementById('map');
const ctx  = cv.getContext('2d');

let MAP_W = {{MAP_W}};
let MAP_H = {{MAP_H}};
let pixBuf  = null;
let imgData = null;

function initBuf(W, H) {
  MAP_W = W; MAP_H = H;
  cv.width  = W;
  cv.height = H;
  pixBuf  = new Uint8ClampedArray(W * H * 4);
  imgData = new ImageData(pixBuf, W, H);
  // fill with unseen colour
  for (let i = 0; i < W * H; i++) {
    pixBuf[i*4]   = 10;
    pixBuf[i*4+1] = 10;
    pixBuf[i*4+2] = 10;
    pixBuf[i*4+3] = 255;
  }
  fitCanvas();
}

function fitCanvas() {
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  const sc = Math.min(cw / MAP_W, ch / MAP_H);
  cv.style.width  = Math.floor(MAP_W * sc) + 'px';
  cv.style.height = Math.floor(MAP_H * sc) + 'px';
}
window.addEventListener('resize', fitCanvas);
initBuf(MAP_W, MAP_H);

function valToRGB(v) {
  if (v === 0)   return [10,  10,  10];    // unseen: near-black
  if (v === 255) return [255, 255, 255];   // wall: white
  return [50, 55, 50];                     // free: dark green-grey
}

let lastPath  = [];
let lastRmx   = MAP_W/2, lastRmy = MAP_H/2, lastTheta = 0;

function applyAndDraw(d) {
  const rows = d.rows, cols = d.cols, vals = d.vals;
  for (let i = 0; i < rows.length; i++) {
    const idx = rows[i] * MAP_W + cols[i];
    const [r,g,b] = valToRGB(vals[i]);
    pixBuf[idx*4]   = r;
    pixBuf[idx*4+1] = g;
    pixBuf[idx*4+2] = b;
    // alpha already 255 from initBuf
  }
  if (d.path)   lastPath  = d.path;
  if (d.rmx != null) { lastRmx = d.rmx; lastRmy = d.rmy; lastTheta = d.rtheta; }

  // Put pixels
  ctx.putImageData(imgData, 0, 0);

  // Path
  if (lastPath.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = '#00ee44';
    ctx.lineWidth = 1.2;
    ctx.lineJoin  = 'round';
    ctx.moveTo(lastPath[0][0]+.5, lastPath[0][1]+.5);
    for (let i = 1; i < lastPath.length; i++)
      ctx.lineTo(lastPath[i][0]+.5, lastPath[i][1]+.5);
    ctx.stroke();
  }

  // Robot arrow
  const R = 6;
  ctx.save();
  ctx.translate(lastRmx+.5, lastRmy+.5);
  ctx.rotate(-lastTheta);
  ctx.beginPath();
  ctx.moveTo(R*1.8, 0);
  ctx.lineTo(-R, R*.85);
  ctx.lineTo(-R*.3, 0);
  ctx.lineTo(-R, -R*.85);
  ctx.closePath();
  ctx.fillStyle   = '#ff3333';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth   = 0.8;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

const socket = io();

socket.on('connect', () => {
  document.getElementById('status').textContent = 'Connected – requesting map…';
  socket.emit('request_full', {});
});
socket.on('disconnect', () => {
  document.getElementById('status').textContent = '⚠ Disconnected – retrying…';
});

socket.on('map_update', function(d) {
  if (d.map_w !== MAP_W || d.map_h !== MAP_H) initBuf(d.map_w, d.map_h);

  applyAndDraw(d);

  const p = d.pose;
  document.getElementById('status').textContent =
    `Scans: ${d.scans} | Pts: ${d.pts} | ` +
    `x=${p.x.toFixed(2)}m  y=${p.y.toFixed(2)}m  θ=${(p.theta*180/Math.PI).toFixed(1)}°`;

  const lEl = document.getElementById('pill-lidar');
  lEl.className   = 'pill ' + (d.lidar   ? 'ok' : 'err');
  lEl.textContent = d.lidar   ? 'Lidar ✓' : 'Lidar ✗';
  const aEl = document.getElementById('pill-ard');
  aEl.className   = 'pill ' + (d.arduino ? 'ok' : 'err');
  aEl.textContent = d.arduino ? 'Arduino ✓' : 'Arduino ✗';
});

socket.on('save_status', function(d) {
  const el = document.getElementById('save-msg');
  el.textContent   = d.msg;
  el.style.color   = d.ok ? '#8f8' : '#f88';
  el.style.display = 'block';
  clearTimeout(el._t);
  el._t = setTimeout(() => el.style.display='none', 5000);
});

function saveMap()  { socket.emit('save_map', {}); }
function resetMap() { if (confirm('Reset the map?')) socket.emit('reset_map', {}); }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML, MAP_W=MAP_W, MAP_H=MAP_H)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Arduino odometry
    threading.Thread(target=arduino_thread, daemon=True, name="arduino").start()

    # Lidar with real-time callback
    try:
        from lidar_ld06 import LD06Driver
        lidar = LD06Driver(port=LIDAR_PORT, baud_rate=LIDAR_BAUD, scan_callback=on_scan)
        lidar.start()
        with _lock:
            _stats["lidar_ok"] = True
        print(f"[INFO] Lidar started on {LIDAR_PORT}")
    except Exception as e:
        print(f"[ERROR] Lidar failed to start: {e}")
        print(f"        Check port: ls /dev/ttyUSB*")

    # Map sender
    threading.Thread(target=sender_thread, daemon=True, name="sender").start()

    print(f"[INFO] Open  http://0.0.0.0:5000  in your browser")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)