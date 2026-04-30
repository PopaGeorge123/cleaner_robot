# VacuumBot — LD06 + Arduino L298 Differential Drive Vacuum Robot

A full-stack robot control system running on Raspberry Pi.

## Hardware
| Component | Port |
|-----------|------|
| Arduino (ROSArduinoBridge) | `/dev/ttyUSB0` |
| LD06 LIDAR | `/dev/ttyUSB1` |

## Architecture

```
Raspberry Pi
├── backend/main.py          Flask + WebSocket server (port 5000)
│   ├── ArduinoDriver        Serial → ROSArduinoBridge (57600 baud)
│   │   ├── set_motor_speeds()   Send "m L R\r" ticks/frame
│   │   ├── read_encoders()      Send "e\r" → "L R"
│   │   └── update_odometry()    Integrate ΔL, ΔR → pose
│   ├── LD06Driver           Serial 230400 baud packet parser
│   │   ├── _parse_packet()  47-byte packets, CRC8, 12pts/pkt
│   │   └── _update_map()    Bresenham ray-cast → occupancy grid
│   ├── Navigator            20 Hz P-controller, heading + dist
│   └── generate_zigzag_plan() Boustrophedon over free cells
└── frontend/index.html      Single-file web app
    ├── WebSocket client     10 Hz state + 2 Hz map updates
    ├── Canvas renderer      Offscreen bitmap + robot overlay
    ├── D-pad + keyboard     WASD / arrow keys
    └── Click-to-navigate    Map click → send goal
```

## Quick Start

```bash
# 1. Clone / copy files onto your Pi
scp -r vacuumbot/ pi@<PI_IP>:/home/pi/

# 2. Run the installer (as root)
sudo bash /home/pi/vacuumbot/setup.sh

# 3. Open web browser
http://<PI_IP>:5000
```

## Manual run (dev)

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Workflow

### 1. Map the area
1. Click **▶ Start Mapping**
2. Drive the robot around all areas using the D-pad (or WASD)
3. Watch the map build in real time
4. When done, click **■ Save Map**

### 2. Navigate to a point
- Click any free (green) cell on the map
- The robot drives there autonomously

### 3. Start cleaning
- Click **⬛ Start Zigzag Clean**
- The robot runs a boustrophedon path over all mapped free space
- Cleaned cells turn blue on the map

## PID Tuning
Default values in `ROSArduinoBridge.ino`:
```
Kp = 20, Kd = 12, Ki = 0, Ko = 50
```
Send `u Kp:Kd:Ki:Ko\r` over serial to update live.

## Robot Parameters (edit top of main.py)
```python
WHEEL_DIAMETER_M  = 0.065   # measure your wheel
WHEEL_BASE_M      = 0.150   # measure track width
ENCODER_TICKS_REV = 1440    # check encoder spec
ZIGZAG_STRIP_WIDTH_M = 0.20 # cleaning lane width
```

## Map File
Saved to `/var/lib/vacuumbot/map.json` and auto-loaded on restart.

## Logs
```bash
journalctl -fu vacuumbot          # live logs
tail -f /var/log/vacuumbot.log    # file log
```

## Keyboard shortcuts (web UI)
| Key | Action |
|-----|--------|
| W / ↑ | Forward |
| S / ↓ | Backward |
| A / ← | Turn left |
| D / → | Turn right |
| Space | Stop |
| Right-click map | Cancel goal |
| Scroll wheel | Zoom map |
| Drag map | Pan view |
