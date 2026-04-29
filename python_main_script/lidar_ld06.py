
# ── LD06Driver: Lidar driver for LD06 2D Lidar ───────────────────────────────
import serial

class LD06Driver:
    """
    LD06 Lidar driver. Reads scans in a background thread and calls scan_callback(scan).
    scan_callback receives a list of (angle_deg, distance_m, intensity) tuples.
    """
    def __init__(self, port, baud_rate=230400, scan_callback=None):
        self.port = port
        self.baud = baud_rate
        self.scan_callback = scan_callback
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _reader(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"[LD06] Connected on {self.port}")
        except Exception as e:
            print(f"[LD06] Serial error: {e}")
            return
        buf = bytearray()
        while self._running:
            try:
                data = ser.read(120)
                if not data:
                    continue
                buf.extend(data)
                # LD06 packet: 47 bytes, starts with 0x54 0x2C
                while len(buf) >= 47:
                    if buf[0] != 0x54 or buf[1] != 0x2C:
                        buf.pop(0)
                        continue
                    pkt = buf[:47]
                    buf = buf[47:]
                    scan = self._parse_packet(pkt)
                    if scan and self.scan_callback:
                        self.scan_callback(scan)
            except Exception as e:
                print(f"[LD06] Read error: {e}")
                time.sleep(0.1)

    def _parse_packet(self, pkt):
        # LD06 packet: 47 bytes, 12 points per packet
        if len(pkt) != 47:
            return None
        if pkt[0] != 0x54 or pkt[1] != 0x2C:
            return None
        # Angle of first point
        fsa = ((pkt[2] | (pkt[3] << 8)) / 100.0)
        # Angle of last point
        lsa = ((pkt[4] | (pkt[5] << 8)) / 100.0)
        # 12 points
        scan = []
        for i in range(12):
            di = 6 + i*3
            dist = (pkt[di] | (pkt[di+1] << 8)) / 1000.0  # meters
            intensity = pkt[di+2]
            if lsa >= fsa:
                angle = fsa + (lsa - fsa) * i / 11
            else:
                angle = fsa + ((lsa + 360) - fsa) * i / 11
                if angle >= 360:
                    angle -= 360
            scan.append((angle, dist, intensity))
        return scan