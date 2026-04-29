"""
live_mapper.py
──────────────
Live mapping and robot trajectory visualization server for LD06 + odometry robots.
- Runs on Raspberry Pi
- Streams map and robot path to any browser in the LAN (WiFi/ethernet)
- No ROS required

Requirements:
    pip install flask flask-socketio eventlet numpy pillow

Usage:
    python3 live_mapper.py
    # Then open http://<raspberry-pi-ip>:5000 in your browser

Robot code dependencies:
    - lidar_ld06.py
    - arduino_driver.py
    - map_loader.py

You can drive the robot with keyboard teleop or any other method.
This script only listens and visualizes.
"""

import threading
import time
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from PIL import Image

from lidar_ld06 import LD06Driver
from arduino_driver import ArduinoDriver
from map_loader import OccupancyGrid
from robot_config import (
    LIDAR_SERIAL_PORT, LIDAR_BAUD_RATE,
    ARDUINO_SERIAL_PORT,
    MAP_RESOLUTION,
)

# ── Flask app setup ───────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── Map and robot state ───────────────────────────────────────────
MAP_W, MAP_H = 200, 200  # 10m x 10m at 0.05m/px
MAP_ORIGIN_X, MAP_ORIGIN_Y = 5.0, 5.0  # meters (centered)
map_grid = np.zeros((MAP_H, MAP_W), dtype=np.uint8)  # 0=free, 100=occupied
robot_path = []  # list of (x, y) in meters

# ── HTML template ─────────────────────────────────────────────────
HTML = '''
<!DOCTYPE html>
<html><head>
<title>Live Robot Mapper</title>
<style>
body { background: #222; color: #eee; font-family: sans-serif; }
canvas { background: #111; border: 2px solid #444; }
</style>
</head><body>
<h2>Live Robot Mapper</h2>
<canvas id="map" width="600" height="600"></canvas>
<br><button onclick="saveMap()" style="font-size:1.2em;padding:0.5em 1em;margin-top:1em;">Salvează harta</button>
<span id="saveStatus" style="margin-left:2em;"></span>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
<script>
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const scale = W / {{MAP_W}};
function draw(map, path) {
  // Draw map
  const img = ctx.createImageData({{MAP_W}}, {{MAP_H}});
  for (let i = 0; i < map.length; ++i) {
    const v = map[i];
    img.data[i*4+0] = v;  // grayscale
    img.data[i*4+1] = v;
    img.data[i*4+2] = v;
    img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  ctx.save();
  ctx.scale(scale, scale);
  // Draw path
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2/scale;
  ctx.beginPath();
  for (let i = 0; i < path.length; ++i) {
    const [x, y] = path[i];
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}
const socket = io();
socket.on('map_update', function(data) {
  draw(new Uint8ClampedArray(data.map), data.path);
});

function saveMap() {
    document.getElementById('saveStatus').textContent = 'Se salvează...';
    socket.emit('save_map', {});
}
socket.on('save_status', function(data) {
    document.getElementById('saveStatus').textContent = data.msg;
});
</script>
</body></html>
'''

@app.route('/')
def index():
    return render_template_string(HTML, MAP_W=MAP_W, MAP_H=MAP_H)

# ── Mapping thread ────────────────────────────────────────────────
def mapping_thread():
    lidar = LD06Driver(port=LIDAR_SERIAL_PORT, baud_rate=LIDAR_BAUD_RATE)
    arduino = ArduinoDriver(port=ARDUINO_SERIAL_PORT)
    lidar.start()
    arduino.start()
    global map_grid, robot_path
    while True:
        # Get robot pose
        pose = arduino.odometry
        x, y, theta = pose.x, pose.y, pose.theta
        robot_path.append((int(MAP_ORIGIN_X / MAP_RESOLUTION + x / MAP_RESOLUTION),
                           int(MAP_ORIGIN_Y / MAP_RESOLUTION - y / MAP_RESOLUTION)))
        # Get lidar scan
        scan = lidar.get_scan()
        # Fade old points for live effect
        map_grid[map_grid > 0] -= 1
        if scan is not None:
            print(f"[DEBUG] Scan received: {len(scan)} points")
            points_written = 0
            for angle, dist, *_ in scan:
                if dist < 0.05 or dist > 8.0:
                    continue
                # Convert to world coordinates
                lx = x + dist * np.cos(theta + np.deg2rad(angle))
                ly = y + dist * np.sin(theta + np.deg2rad(angle))
                mx = int(MAP_ORIGIN_X / MAP_RESOLUTION + lx / MAP_RESOLUTION)
                my = int(MAP_ORIGIN_Y / MAP_RESOLUTION - ly / MAP_RESOLUTION)
                if 0 <= mx < MAP_W and 0 <= my < MAP_H:
                    map_grid[my, mx] = 255
                    points_written += 1
            print(f"[DEBUG] Points written to map: {points_written}")
        else:
            print("[DEBUG] No scan received")
        # Send to clients
        socketio.emit('map_update', {
            'map': map_grid.flatten().tolist(),
            'path': robot_path[-500:],
        })
        time.sleep(0.2)


# ── Save map handler ─────────────────────────────────────────────
from map_loader import OccupancyGrid
@socketio.on('save_map')
def handle_save_map(_):
    global map_grid
    try:
        grid = OccupancyGrid(
            data=map_grid,
            resolution=MAP_RESOLUTION,
            origin=(-MAP_ORIGIN_X, -MAP_ORIGIN_Y, 0.0)
        )
        fname = f"map_{int(time.time())}"
        grid.save(fname)
        socketio.emit('save_status', {'msg': f'Harta a fost salvată: {fname}'})
    except Exception as e:
        socketio.emit('save_status', {'msg': f'Eroare la salvare: {e}'})


# ── Main entrypoint ─────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=mapping_thread, daemon=True)
    t.start()
    socketio.run(app, host="0.0.0.0", port=5000)
