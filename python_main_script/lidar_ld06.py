"""
live_mapper.py  (v4 - binary transfer + browser keyboard control)
──────────────────────────────────────────────────────────────────
Fixes vs v3:
  - Map sent as raw binary (base64) not JSON arrays → 30x smaller payload
  - Only the dirty rectangle is sent each frame, not the whole map
  - Browser WASD keyboard drives the robot in real time via SocketIO
  - Reverse works: negative encoder ticks = robot moves backward on map
  - Odometry runs in separate thread talking directly to Arduino serial
  - Lidar callback paints map immediately (no polling)
  - Clean UI with speed slider, emergency stop button

Requirements:
    pip install flask flask-socketio eventlet numpy pillow pyserial

Run:
    python3 live_mapper.py
    Open http://<pi-ip>:5000  and use WASD to drive
"""

import base64
import math
import struct
import threading
import time
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ── Config ────────────────────────────────────────────────────────────────────
LIDAR_PORT   = "/dev/ttyUSB1"
LIDAR_BAUD   = 230400
ARDUINO_PORT = "/dev/ttyUSB0"
ARDUINO_BAUD = 57600
MAP_RES      = 0.05    # metres per pixel
MAP_WORLD_M  = 20.0    # 20 m × 20 m world
SEND_HZ      = 10      # browser refresh rate

try:
    from robot_config import (
        LIDAR_SERIAL_PORT  as LIDAR_PORT,
        LIDAR_BAUD_RATE    as LIDAR_BAUD,
        ARDUINO_SERIAL_PORT as ARDUINO_PORT,
        ARDUINO_BAUD_RATE  as ARDUINO_BAUD,
        MAP_RESOLUTION     as MAP_RES,
    )
except Exception:
    pass

MAP_W  = int(MAP_WORLD_M / MAP_RES)
MAP_H  = int(MAP_WORLD_M / MAP_RES)
MAP_CX = MAP_W // 2
MAP_CY = MAP_H // 2

# metres/tick and wheel base — overridden by robot_config if present
try:
    from robot_config import METRES_PER_TICK, WHEEL_SEPARATION
except Exception:
    METRES_PER_TICK  = (2 * math.pi * 0.033) / 3436
    WHEEL_SEPARATION = 0.297

try:
    from robot_config import MAX_LINEAR_VEL, MAX_ANGULAR_VEL
except Exception:
    MAX_LINEAR_VEL  = 0.22
    MAX_ANGULAR_VEL = 1.0

PID_RATE = 30   # Hz — must match Arduino firmware

# ── Shared state ──────────────────────────────────────────────────────────────
_lock      = threading.Lock()
_map       = np.zeros((MAP_H, MAP_W), dtype=np.uint8)
# dirty region tracking
_dirty     = np.zeros((MAP_H, MAP_W), dtype=bool)
_path_px   = []                     # [(mx, my), ...]
_pose      = [0.0, 0.0, 0.0]        # x, y, theta  (m, m, rad)
_stats     = {"scans": 0, "pts": 0, "lidar_ok": False, "arduino_ok": False}
_reset_evt = threading.Event()

# Velocity command from browser (ticks per PID frame, signed)
_cmd_lock = threading.Lock()
_cmd      = [0, 0]    # [left_tpf, right_tpf]

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet",
                    max_http_buffer_size=1_000_000)


# ── Helpers ───────────────────────────────────────────────────────────────────
def w2m(wx, wy):
    """World metres → map pixel (col=mx, row=my). +Y world = up = lower row."""
    mx = int(MAP_CX + wx / MAP_RES)
    my = int(MAP_CY - wy / MAP_RES)
    return mx, my


def _bresenham(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cx, cy = x0, y0
    for _ in range(800):           # cap at 800 steps ≈ 40 m
        pts.append((cx, cy))
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 > -dy: err -= dy; cx += sx
        if e2 <  dx: err += dx; cy += sy
    return pts


# ── Lidar callback (fires from LD06 reader thread, ~10 Hz) ───────────────────
def on_scan(scan):
    with _lock:
        x, y, theta = _pose[0], _pose[1], _pose[2]
    rmx, rmy = w2m(x, y)

    with _lock:
        for angle_deg, dist_m, _ in scan:
            if dist_m < 0.05 or dist_m > 10.0:
                continue
            a   = theta + math.radians(angle_deg)
            ox  = x + dist_m * math.cos(a)
            oy  = y + dist_m * math.sin(a)
            omx, omy = w2m(ox, oy)

            # Paint wall
            if 0 <= omx < MAP_W and 0 <= omy < MAP_H:
                _map[omy, omx] = 255
                _dirty[omy, omx] = True

            # Bresenham free-space ray (skip last cell = wall)
            for bx, by in _bresenham(rmx, rmy, omx, omy)[:-1]:
                if 0 <= bx < MAP_W and 0 <= by < MAP_H and _map[by, bx] < 255:
                    if _map[by, bx] < 60:
                        _map[by, bx] = 60
                        _dirty[by, bx] = True

        _stats["scans"] += 1
        _stats["pts"]   += len(scan)
        _stats["lidar_ok"] = True


# ── Arduino thread: odometry + motor commands ─────────────────────────────────
def arduino_thread():
    """
    Reads encoder ticks at 30 Hz → updates _pose.
    Sends motor speed commands from _cmd at 30 Hz.
    Encoder deltas are signed → reverse works automatically.
    """
    try:
        import serial
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.3)
        time.sleep(2.0)
        ser.reset_input_buffer()
        # Reset Arduino encoders
        ser.write(b"r\r")
        ser.readline()
        with _lock:
            _stats["arduino_ok"] = True
        print(f"[Arduino] Connected on {ARDUINO_PORT}")
    except Exception as e:
        print(f"[Arduino] FAILED: {e}  —  check port with: ls /dev/ttyUSB*")
        return

    def send(cmd: str) -> str:
        try:
            ser.reset_input_buffer()
            ser.write((cmd + "\r").encode())
            return ser.readline().decode(errors="replace").strip()
        except Exception:
            return ""

    prev_l = prev_r = 0
    interval = 1.0 / PID_RATE

    while True:
        t0 = time.monotonic()

        # --- Send velocity command ---
        with _cmd_lock:
            cl, cr = _cmd[0], _cmd[1]
        send(f"m {cl} {cr}")

        # --- Read encoders ---
        raw = send("e")
        parts = raw.split()
        if len(parts) >= 2:
            try:
                nl, nr = int(parts[0]), int(parts[1])
                # Signed deltas — negative = reverse, positive = forward
                dl = nl - prev_l
                dr = nr - prev_r
                prev_l, prev_r = nl, nr

                dist_l  = dl * METRES_PER_TICK
                dist_r  = dr * METRES_PER_TICK
                dist    = (dist_l + dist_r) / 2.0
                dtheta  = (dist_r - dist_l) / WHEEL_SEPARATION

                with _lock:
                    _pose[2] += dtheta
                    mid_theta = _pose[2] - dtheta / 2.0
                    _pose[0] += dist * math.cos(mid_theta)
                    _pose[1] += dist * math.sin(mid_theta)
                    # Wrap theta to [-pi, pi]
                    while _pose[2] >  math.pi: _pose[2] -= 2 * math.pi
                    while _pose[2] < -math.pi: _pose[2] += 2 * math.pi

                    mx, my = w2m(_pose[0], _pose[1])
                    if 0 <= mx < MAP_W and 0 <= my < MAP_H:
                        if not _path_px or _path_px[-1] != (mx, my):
                            _path_px.append((mx, my))
                            if len(_path_px) > 8000:
                                del _path_px[:2000]
            except ValueError:
                pass

        elapsed = time.monotonic() - t0
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ── Map sender thread ─────────────────────────────────────────────────────────
def sender_thread():
    """
    Sends map updates to browser at SEND_HZ.
    Only the dirty bounding rectangle is sent as binary (base64).
    This is ~30× smaller than JSON pixel arrays.
    """
    interval = 1.0 / SEND_HZ

    while True:
        t0 = time.monotonic()

        # Handle reset
        if _reset_evt.is_set():
            _reset_evt.clear()
            with _lock:
                _map[:] = 0
                _dirty[:] = False
                _path_px.clear()
                _pose[0] = _pose[1] = _pose[2] = 0.0
                _stats["scans"] = _stats["pts"] = 0
            socketio.emit('map_reset', {'map_w': MAP_W, 'map_h': MAP_H})
            print("[INFO] Map reset")

        with _lock:
            dirty_rows, dirty_cols = np.where(_dirty)
            if len(dirty_rows) == 0:
                # Nothing changed — just send pose update
                px, py, ptheta = _pose
                sc = dict(_stats)
                mx, my = w2m(px, py)
                socketio.emit('pose_update', {
                    'rmx': mx, 'rmy': my, 'rtheta': ptheta,
                    'pose': {'x': round(px,3), 'y': round(py,3), 'theta': round(ptheta,3)},
                    'path': list(_path_px[-500:]),
                    'scans': sc['scans'], 'pts': sc['pts'],
                    'lidar': sc['lidar_ok'], 'arduino': sc['arduino_ok'],
                })
            else:
                # Dirty bounding box
                r0 = int(dirty_rows.min()); r1 = int(dirty_rows.max()) + 1
                c0 = int(dirty_cols.min()); c1 = int(dirty_cols.max()) + 1
                patch = _map[r0:r1, c0:c1].copy()
                _dirty[:] = False

                px, py, ptheta = _pose
                sc = dict(_stats)
                mx, my = w2m(px, py)

        if len(dirty_rows) > 0:
            # Encode patch as binary base64 — tiny payload
            raw_bytes = patch.tobytes()                     # uint8 flat
            b64       = base64.b64encode(raw_bytes).decode()

            socketio.emit('map_patch', {
                'r0': r0, 'c0': c0,
                'rows': r1 - r0, 'cols': c1 - c0,
                'data': b64,                                # base64 uint8 patch
                'rmx': mx, 'rmy': my, 'rtheta': ptheta,
                'pose': {'x': round(px,3), 'y': round(py,3), 'theta': round(ptheta,3)},
                'path': list(_path_px[-500:]),
                'scans': sc['scans'], 'pts': sc['pts'],
                'lidar': sc['lidar_ok'], 'arduino': sc['arduino_ok'],
            })

        elapsed = time.monotonic() - t0
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on('request_full')
def handle_full(_):
    """Send entire map as one binary patch on (re)connect."""
    with _lock:
        snap      = _map.copy()
        path_snap = list(_path_px[-500:])
        px, py, ptheta = _pose
        sc = dict(_stats)
    mx, my = w2m(px, py)
    b64 = base64.b64encode(snap.tobytes()).decode()
    socketio.emit('map_patch', {
        'r0': 0, 'c0': 0,
        'rows': MAP_H, 'cols': MAP_W,
        'data': b64,
        'rmx': mx, 'rmy': my, 'rtheta': ptheta,
        'pose': {'x': round(px,3), 'y': round(py,3), 'theta': round(ptheta,3)},
        'path': path_snap,
        'scans': sc['scans'], 'pts': sc['pts'],
        'lidar': sc['lidar_ok'], 'arduino': sc['arduino_ok'],
    })


@socketio.on('cmd_vel')
def handle_cmd_vel(data):
    """
    Browser sends: {'linear': m/s, 'angular': rad/s}
    Converted to ticks/frame and stored in _cmd for arduino_thread to send.
    """
    linear  = float(data.get('linear',  0.0))
    angular = float(data.get('angular', 0.0))
    linear  = max(-MAX_LINEAR_VEL,  min(MAX_LINEAR_VEL,  linear))
    angular = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, angular))
    v_l = linear - angular * WHEEL_SEPARATION / 2.0
    v_r = linear + angular * WHEEL_SEPARATION / 2.0
    tpf_l = round(v_l / METRES_PER_TICK / PID_RATE)
    tpf_r = round(v_r / METRES_PER_TICK / PID_RATE)
    with _cmd_lock:
        _cmd[0] = tpf_l
        _cmd[1] = tpf_r


@socketio.on('cmd_stop')
def handle_stop(_):
    with _cmd_lock:
        _cmd[0] = _cmd[1] = 0


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

        # PGM: black=occupied, white=free, 205=unknown  (ROS convention)
        pgm = np.full((MAP_H, MAP_W), 205, dtype=np.uint8)
        pgm[snap == 255] = 0
        pgm[(snap > 0) & (snap < 255)] = 254

        with open(fname + ".pgm", "wb") as f:
            f.write(f"P5\n{MAP_W} {MAP_H}\n255\n".encode())
            f.write(pgm.tobytes())

        ox = -MAP_CX * MAP_RES
        oy = -MAP_CY * MAP_RES
        with open(fname + ".yaml", "w") as f:
            f.write(
                f"image: {fname}.pgm\n"
                f"resolution: {MAP_RES}\n"
                f"origin: [{ox:.4f}, {oy:.4f}, 0.0]\n"
                f"negate: 0\n"
                f"occupied_thresh: 0.65\n"
                f"free_thresh: 0.196\n"
            )

        msg = f"✓ Saved {fname}.pgm + .yaml"
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(pgm).save(fname + ".png")
            msg += " + .png"
        except ImportError:
            pass

        print(f"[INFO] {msg}")
        socketio.emit('save_status', {'ok': True, 'msg': msg})
    except Exception as e:
        socketio.emit('save_status', {'ok': False, 'msg': f'✗ {e}'})


# ── HTML + JS ─────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Live Robot Mapper</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#ddd;font-family:monospace;display:flex;flex-direction:column;height:100vh;overflow:hidden;user-select:none}

/* ── Header ── */
#hdr{display:flex;align-items:center;gap:8px;padding:5px 10px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;flex-shrink:0;flex-wrap:wrap}
#hdr h2{font-size:13px;color:#fff;white-space:nowrap;margin-right:4px}
.pill{padding:2px 7px;border-radius:9px;font-size:10px;background:#282828;color:#666;white-space:nowrap}
.pill.ok{background:#0d2e0d;color:#3d3}
.pill.err{background:#2e0d0d;color:#d33}
#status{font-size:10px;color:#888;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.hbtn{background:#1e4d2e;color:#6c6;border:1px solid #2a6a3a;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer;white-space:nowrap}
.hbtn:hover{background:#2a6a3a}
.hbtn.red{background:#3a1010;color:#c55;border-color:#6a2020}
.hbtn.red:hover{background:#5a1515}

/* ── Main area ── */
#main{flex:1;display:flex;overflow:hidden}

/* ── Map panel ── */
#map-wrap{flex:1;display:flex;align-items:center;justify-content:center;background:#0a0a0a;position:relative;overflow:hidden}
canvas{image-rendering:pixelated;image-rendering:crisp-edges;display:block;cursor:crosshair}
#save-msg{position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.9);font-size:12px;padding:6px 18px;border-radius:6px;display:none;pointer-events:none;white-space:nowrap}
#legend{position:absolute;top:8px;left:8px;background:rgba(0,0,0,.7);padding:6px 10px;border-radius:5px;font-size:10px;line-height:2;color:#aaa}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:middle}

/* ── Control panel ── */
#ctrl{width:200px;flex-shrink:0;background:#151515;border-left:1px solid #222;display:flex;flex-direction:column;gap:0;padding:0}
.ctrl-sec{padding:10px 12px;border-bottom:1px solid #222}
.ctrl-sec h3{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}

/* WASD pad */
#dpad{display:grid;grid-template-columns:repeat(3,44px);grid-template-rows:repeat(3,44px);gap:4px;justify-content:center}
.dkey{background:#222;border:1px solid #333;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px;cursor:pointer;transition:background .1s;color:#888}
.dkey:active,.dkey.active{background:#1e4d2e;border-color:#2a6;color:#4c4}
.dkey.stop-btn{background:#2a1515;border-color:#6a2020;color:#c55;font-size:12px}
.dkey.stop-btn:active,.dkey.stop-btn.active{background:#5a1515}

/* Speed slider */
#speed-val{color:#4c4;font-size:13px;font-weight:bold}
input[type=range]{width:100%;accent-color:#2a6}

/* Info */
.info-row{display:flex;justify-content:space-between;font-size:10px;color:#666;margin-bottom:3px}
.info-val{color:#aaa}

/* Keyboard hint */
.hint{font-size:9px;color:#444;text-align:center;padding:6px;line-height:1.6}
</style>
</head>
<body>

<div id="hdr">
  <h2>🤖 Robot Mapper</h2>
  <span id="pill-lidar"  class="pill">Lidar ✗</span>
  <span id="pill-ard"    class="pill">Arduino ✗</span>
  <div id="status">Connecting…</div>
  <button class="hbtn" onclick="saveMap()">💾 Save</button>
  <button class="hbtn red" onclick="resetMap()">🗑 Reset</button>
</div>

<div id="main">
  <!-- Map -->
  <div id="map-wrap">
    <canvas id="map"></canvas>
    <div id="legend">
      <span class="dot" style="background:#fff"></span>Wall<br>
      <span class="dot" style="background:#3a3a3a;border:1px solid #555"></span>Free<br>
      <span class="dot" style="background:#0a0a0a;border:1px solid #333"></span>Unseen<br>
      <span class="dot" style="background:#0c0;border-radius:50%"></span>Path<br>
      <span class="dot" style="background:#f44;border-radius:50%"></span>Robot
    </div>
    <div id="save-msg"></div>
  </div>

  <!-- Control panel -->
  <div id="ctrl">
    <div class="ctrl-sec">
      <h3>Drive (WASD / click)</h3>
      <div id="dpad">
        <div></div>
        <div class="dkey" id="btn-w" onmousedown="kdown('w')" onmouseup="kup('w')" ontouchstart="kdown('w')" ontouchend="kup('w')">▲</div>
        <div></div>
        <div class="dkey" id="btn-a" onmousedown="kdown('a')" onmouseup="kup('a')" ontouchstart="kdown('a')" ontouchend="kup('a')">◀</div>
        <div class="dkey stop-btn" id="btn-spc" onmousedown="emergStop()" ontouchstart="emergStop()">■<br><span style="font-size:8px">STOP</span></div>
        <div class="dkey" id="btn-d" onmousedown="kdown('d')" onmouseup="kup('d')" ontouchstart="kdown('d')" ontouchend="kup('d')">▶</div>
        <div></div>
        <div class="dkey" id="btn-s" onmousedown="kdown('s')" onmouseup="kup('s')" ontouchstart="kdown('s')" ontouchend="kup('s')">▼</div>
        <div></div>
      </div>
    </div>

    <div class="ctrl-sec">
      <h3>Speed  <span id="speed-val">50%</span></h3>
      <input type="range" id="speed-slider" min="10" max="100" value="50" oninput="onSpeedChange(this.value)">
    </div>

    <div class="ctrl-sec">
      <h3>Status</h3>
      <div class="info-row"><span>Scans</span><span class="info-val" id="inf-scans">0</span></div>
      <div class="info-row"><span>Points</span><span class="info-val" id="inf-pts">0</span></div>
      <div class="info-row"><span>X</span><span class="info-val" id="inf-x">0.00 m</span></div>
      <div class="info-row"><span>Y</span><span class="info-val" id="inf-y">0.00 m</span></div>
      <div class="info-row"><span>θ</span><span class="info-val" id="inf-t">0.0°</span></div>
    </div>

    <div class="hint">
      Keyboard: W A S D<br>
      Space = stop<br>
      Click map to focus
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
<script>
// ── Canvas setup ──────────────────────────────────────────────────────────────
const wrap = document.getElementById('map-wrap');
const cv   = document.getElementById('map');
const ctx  = cv.getContext('2d');
let MAP_W  = {{MAP_W}};
let MAP_H  = {{MAP_H}};
let pixBuf = null;
let imgData= null;

function initBuf(W, H) {
  MAP_W = W; MAP_H = H;
  cv.width = W; cv.height = H;
  pixBuf  = new Uint8ClampedArray(W * H * 4);
  imgData = new ImageData(pixBuf, W, H);
  for (let i = 0; i < W*H; i++) {
    pixBuf[i*4]=10; pixBuf[i*4+1]=10; pixBuf[i*4+2]=10; pixBuf[i*4+3]=255;
  }
  fitCanvas();
}
function fitCanvas() {
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  const sc = Math.min(cw/MAP_W, ch/MAP_H);
  cv.style.width  = Math.floor(MAP_W*sc)+'px';
  cv.style.height = Math.floor(MAP_H*sc)+'px';
}
window.addEventListener('resize', fitCanvas);
initBuf(MAP_W, MAP_H);

function valToRGB(v) {
  if (v === 0)   return [10,  10,  10];
  if (v === 255) return [240, 240, 240];
  return [50, 58, 50];   // free: dark green-grey
}

// Apply binary patch (base64 uint8 array → pixBuf)
function applyPatch(r0, c0, rows, cols, b64) {
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  let i = 0;
  for (let r = r0; r < r0+rows; r++) {
    for (let c = c0; c < c0+cols; c++) {
      const v   = raw[i++];
      const idx = (r*MAP_W + c)*4;
      const [rv,gv,bv] = valToRGB(v);
      pixBuf[idx]   = rv;
      pixBuf[idx+1] = gv;
      pixBuf[idx+2] = bv;
    }
  }
}

let lastPath = [], lastRmx = MAP_W/2, lastRmy = MAP_H/2, lastTheta = 0;

function redraw() {
  ctx.putImageData(imgData, 0, 0);

  // Path
  if (lastPath.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = '#00dd44';
    ctx.lineWidth   = 1.2;
    ctx.lineJoin    = 'round';
    ctx.moveTo(lastPath[0][0]+.5, lastPath[0][1]+.5);
    for (let i=1;i<lastPath.length;i++) ctx.lineTo(lastPath[i][0]+.5, lastPath[i][1]+.5);
    ctx.stroke();
  }

  // Robot arrow
  const R = 6;
  ctx.save();
  ctx.translate(lastRmx+.5, lastRmy+.5);
  ctx.rotate(-lastTheta);
  ctx.beginPath();
  ctx.moveTo(R*2, 0);
  ctx.lineTo(-R, R*.9);
  ctx.lineTo(-R*.3, 0);
  ctx.lineTo(-R, -R*.9);
  ctx.closePath();
  ctx.fillStyle   = '#ff3333';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth   = 0.6;
  ctx.fill(); ctx.stroke();
  ctx.restore();
}

function updateOverlayOnly(d) {
  // redraw without touching pixBuf (path/robot only)
  if (d.path)   lastPath = d.path;
  if (d.rmx != null) { lastRmx=d.rmx; lastRmy=d.rmy; lastTheta=d.rtheta; }
  redraw();
}

function updateStatus(d) {
  const p = d.pose || {};
  document.getElementById('inf-scans').textContent = d.scans ?? '-';
  document.getElementById('inf-pts').textContent   = d.pts   ?? '-';
  document.getElementById('inf-x').textContent     = (p.x ?? 0).toFixed(3)+' m';
  document.getElementById('inf-y').textContent     = (p.y ?? 0).toFixed(3)+' m';
  document.getElementById('inf-t').textContent     = ((p.theta??0)*180/Math.PI).toFixed(1)+'°';
  document.getElementById('status').textContent    =
    `x=${(p.x??0).toFixed(2)}m  y=${(p.y??0).toFixed(2)}m  θ=${((p.theta??0)*180/Math.PI).toFixed(1)}°  |  scans:${d.scans??0}`;

  const lEl=document.getElementById('pill-lidar');
  lEl.className='pill '+(d.lidar?'ok':'err');
  lEl.textContent=d.lidar?'Lidar ✓':'Lidar ✗';
  const aEl=document.getElementById('pill-ard');
  aEl.className='pill '+(d.arduino?'ok':'err');
  aEl.textContent=d.arduino?'Arduino ✓':'Arduino ✗';
}

// ── Socket ────────────────────────────────────────────────────────────────────
const socket = io();
socket.on('connect', () => {
  document.getElementById('status').textContent='Connected – loading map…';
  socket.emit('request_full', {});
});
socket.on('disconnect', () => {
  document.getElementById('status').textContent='⚠ Disconnected…';
});
socket.on('map_patch', d => {
  if (d.map_w && (d.map_w!==MAP_W||d.map_h!==MAP_H)) initBuf(d.map_w, d.map_h);
  applyPatch(d.r0, d.c0, d.rows, d.cols, d.data);
  if (d.path)  lastPath=d.path;
  if (d.rmx!=null){lastRmx=d.rmx;lastRmy=d.rmy;lastTheta=d.rtheta;}
  redraw();
  updateStatus(d);
});
socket.on('pose_update', d => {
  updateOverlayOnly(d);
  updateStatus(d);
});
socket.on('map_reset', d => {
  initBuf(d.map_w, d.map_h);
  lastPath=[]; redraw();
});
socket.on('save_status', d => {
  const el=document.getElementById('save-msg');
  el.textContent=d.msg; el.style.color=d.ok?'#8f8':'#f88';
  el.style.display='block';
  clearTimeout(el._t); el._t=setTimeout(()=>el.style.display='none',5000);
});

// ── Velocity control ──────────────────────────────────────────────────────────
const MAX_LIN = {{MAX_LIN}};
const MAX_ANG = {{MAX_ANG}};
let speedFrac = 0.5;

const keys = {w:false, a:false, s:false, d:false};
const btnMap = {w:'btn-w', a:'btn-a', s:'btn-s', d:'btn-d'};

function kdown(k) { keys[k]=true;  updateBtn(k,true);  sendVel(); }
function kup(k)   { keys[k]=false; updateBtn(k,false); sendVel(); }
function updateBtn(k,on) {
  const el=document.getElementById(btnMap[k]);
  if(el) el.classList.toggle('active',on);
}

document.addEventListener('keydown', e => {
  if(['w','a','s','d',' '].includes(e.key.toLowerCase()||e.key)) e.preventDefault();
  const k = e.key.toLowerCase();
  if(k===' ') { emergStop(); return; }
  if(keys[k]===false) kdown(k);
});
document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  if(keys[k]===true) kup(k);
});

let velTimer = null;
function sendVel() {
  clearInterval(velTimer);
  const lin = (keys.w?1:0)-(keys.s?1:0);
  const ang = (keys.a?1:0)-(keys.d?1:0);
  if(lin===0&&ang===0) {
    socket.emit('cmd_vel',{linear:0,angular:0});
    return;
  }
  // Send repeatedly while key held (Arduino auto-stops after 2 s)
  function doSend() {
    socket.emit('cmd_vel',{
      linear:  lin * MAX_LIN * speedFrac,
      angular: ang * MAX_ANG * speedFrac,
    });
  }
  doSend();
  velTimer = setInterval(doSend, 200);
}

function emergStop() {
  Object.keys(keys).forEach(k=>{ keys[k]=false; updateBtn(k,false); });
  clearInterval(velTimer);
  socket.emit('cmd_stop',{});
}

function onSpeedChange(v) {
  speedFrac = v/100;
  document.getElementById('speed-val').textContent = v+'%';
  if(Object.values(keys).some(Boolean)) sendVel();
}

function saveMap()  { socket.emit('save_map', {}); }
function resetMap() { if(confirm('Reset the map?')) socket.emit('reset_map',{}); }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML,
        MAP_W=MAP_W, MAP_H=MAP_H,
        MAX_LIN=MAX_LINEAR_VEL,
        MAX_ANG=MAX_ANGULAR_VEL,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Arduino: odometry + motor commands
    threading.Thread(target=arduino_thread, daemon=True, name="arduino").start()

    # Lidar: scan callback fires on_scan() from its own reader thread
    try:
        from lidar_ld06 import LD06Driver
        lidar = LD06Driver(port=LIDAR_PORT, baud_rate=LIDAR_BAUD,
                           scan_callback=on_scan)
        lidar.start()
        with _lock:
            _stats["lidar_ok"] = True
        print(f"[INFO] Lidar started on {LIDAR_PORT}")
    except Exception as e:
        print(f"[ERROR] Lidar failed: {e}")
        print(f"        Check port: ls /dev/ttyUSB*")

    # Sender: pushes binary patches to browser at 10 Hz
    threading.Thread(target=sender_thread, daemon=True, name="sender").start()

    print(f"[INFO] Open  http://0.0.0.0:5000  in your browser")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)