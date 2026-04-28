"""
ball_tracker.py
---------------
HSV-based ball detection and follow-ball controller.

Mirrors the behaviour of the ROS `detect_ball` + `follow_ball` nodes from
the articubot_one project – but using pure OpenCV + Python, no ROS.

Parameters come from ball_tracker_params_sim.yaml / robot_config.py.

No ROS required.  Dependencies: numpy, opencv-python.

Usage (live camera)
-------------------
    from ball_tracker import BallDetector, BallFollower
    from diff_drive   import DiffDriveController

    detector = BallDetector()
    follower = BallFollower(DiffDriveController())

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        result = detector.detect(frame)
        v, w = follower.update(result)
        # send v, w to robot hardware

Usage (image file / test)
--------------------------
    result = detector.detect(cv2.imread("ball_photo.jpg"))
    print(result)  # BallDetection(found=True, cx=0.1, cy=0.05, size=0.02)
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from robot_config import (
    BALL_HSV_MIN, BALL_HSV_MAX,
    BALL_SIZE_MIN, BALL_SIZE_MAX,
    ANGULAR_CHASE_MULTIPLIER,
    FORWARD_CHASE_SPEED,
    SEARCH_ANGULAR_SPEED,
    MAX_BALL_SIZE_THRESH,
    FILTER_VALUE,
    MAX_LINEAR_VEL, MAX_ANGULAR_VEL,
)


# ── Detection result ──────────────────────────────────────────────────────────

@dataclass
class BallDetection:
    """
    Result from BallDetector.detect().

    Attributes
    ----------
    found  : bool   – True if a ball blob was found
    cx     : float  – normalised horizontal offset from frame centre  [-0.5, 0.5]
    cy     : float  – normalised vertical   offset from frame centre  [-0.5, 0.5]
    size   : float  – normalised blob area relative to frame area     [0, 1]
    pixel_x: int    – raw pixel x of blob centre (0 if not found)
    pixel_y: int    – raw pixel y of blob centre (0 if not found)
    """
    found:   bool  = False
    cx:      float = 0.0
    cy:      float = 0.0
    size:    float = 0.0
    pixel_x: int   = 0
    pixel_y: int   = 0


# ── Detector ──────────────────────────────────────────────────────────────────

class BallDetector:
    """
    Detect a coloured ball in a BGR image using HSV thresholding.

    Parameters come from ball_tracker_params_sim.yaml.
    """

    def __init__(
        self,
        h_min: int = BALL_HSV_MIN[0],
        h_max: int = BALL_HSV_MAX[0],
        s_min: int = BALL_HSV_MIN[1],
        s_max: int = BALL_HSV_MAX[1],
        v_min: int = BALL_HSV_MIN[2],
        v_max: int = BALL_HSV_MAX[2],
        sz_min: float = BALL_SIZE_MIN,
        sz_max: float = BALL_SIZE_MAX,
        x_min: int = 0,   # ROI crop  (0–100 %)
        x_max: int = 100,
        y_min: int = 32,
        y_max: int = 100,
        tuning_mode: bool = False,
    ):
        self.lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        self.upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
        self.sz_min = sz_min / 100.0
        self.sz_max = sz_max / 100.0
        self.x_min  = x_min  / 100.0
        self.x_max  = x_max  / 100.0
        self.y_min  = y_min  / 100.0
        self.y_max  = y_max  / 100.0
        self.tuning_mode = tuning_mode

    def detect(self, frame: np.ndarray) -> BallDetection:
        """
        Detect the ball in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray  H×W×3 BGR image (as returned by cv2.VideoCapture.read)

        Returns
        -------
        BallDetection
        """
        H, W = frame.shape[:2]

        # Apply ROI crop
        r0 = int(self.y_min * H)
        r1 = int(self.y_max * H)
        c0 = int(self.x_min * W)
        c1 = int(self.x_max * W)
        roi = frame[r0:r1, c0:c1]

        # HSV threshold
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if self.tuning_mode:
            cv2.imshow("HSV mask", mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BallDetection(found=False)

        # Pick the largest contour
        c = max(contours, key=cv2.contourArea)
        area       = cv2.contourArea(c)
        frame_area = roi.shape[0] * roi.shape[1]
        norm_size  = area / frame_area

        if not (self.sz_min <= norm_size <= self.sz_max):
            return BallDetection(found=False)

        # Centroid
        M = cv2.moments(c)
        if M['m00'] == 0:
            return BallDetection(found=False)

        cx_px = int(M['m10'] / M['m00']) + c0
        cy_px = int(M['m01'] / M['m00']) + r0

        cx_norm = (cx_px / W) - 0.5       # [-0.5, 0.5]
        cy_norm = (cy_px / H) - 0.5

        return BallDetection(
            found   = True,
            cx      = cx_norm,
            cy      = cy_norm,
            size    = norm_size,
            pixel_x = cx_px,
            pixel_y = cy_px,
        )

    def annotate(self, frame: np.ndarray, det: BallDetection) -> np.ndarray:
        """Draw detection overlay on a copy of the frame."""
        out = frame.copy()
        if det.found:
            H, W = frame.shape[:2]
            cv2.circle(out, (det.pixel_x, det.pixel_y), 15, (0, 255, 0), 2)
            cv2.line(out, (W // 2, H // 2), (det.pixel_x, det.pixel_y),
                     (0, 255, 0), 1)
            cv2.putText(out, f"size={det.size:.3f}",
                        (det.pixel_x + 10, det.pixel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out


# ── Follower ──────────────────────────────────────────────────────────────────

class BallFollower:
    """
    Generate (linear, angular) velocity commands to chase a detected ball.

    Mirrors the `follow_ball` ROS node logic.

    Parameters
    ----------
    rcv_timeout_secs         : float  – treat ball as lost after this many seconds
    angular_chase_multiplier : float  – how aggressively to steer toward ball
    forward_chase_speed      : float  – m/s while chasing
    search_angular_speed     : float  – rad/s spin when searching
    max_size_thresh          : float  – stop when ball fills this fraction of frame
    filter_value             : float  – low-pass weight (higher = more smoothing)
    """

    def __init__(
        self,
        rcv_timeout_secs:         float = 1.0,
        angular_chase_multiplier: float = ANGULAR_CHASE_MULTIPLIER,
        forward_chase_speed:      float = FORWARD_CHASE_SPEED,
        search_angular_speed:     float = SEARCH_ANGULAR_SPEED,
        max_size_thresh:          float = MAX_BALL_SIZE_THRESH,
        filter_value:             float = FILTER_VALUE,
    ):
        self.timeout      = rcv_timeout_secs
        self.ang_mult     = angular_chase_multiplier
        self.fwd_speed    = forward_chase_speed
        self.search_speed = search_angular_speed
        self.max_size     = max_size_thresh
        self.alpha        = filter_value   # EMA: output = alpha * prev + (1-alpha) * new

        self._last_seen: float = 0.0
        self._cx_filt:   float = 0.0

    def update(self, detection: BallDetection) -> Tuple[float, float]:
        """
        Compute (linear m/s, angular rad/s) command from a BallDetection.

        Returns
        -------
        (v, w)
        """
        now = time.monotonic()

        if detection.found:
            self._last_seen = now

            # Low-pass filter the horizontal offset
            self._cx_filt = (
                self.alpha * self._cx_filt
                + (1.0 - self.alpha) * detection.cx
            )

            # Too close? Stop.
            if detection.size > self.max_size:
                return 0.0, 0.0

            angular = -self.ang_mult * self._cx_filt
            linear  = self.fwd_speed

            angular = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, angular))
            return linear, angular

        # Lost ball – search by spinning
        if (now - self._last_seen) > self.timeout:
            return 0.0, self.search_angular_speed

        return 0.0, 0.0


# ── Live camera runner ────────────────────────────────────────────────────────

def run_live_tracker(
    camera_index: int = 0,
    tuning_mode:  bool = False,
) -> None:
    """
    Open webcam and display real-time ball detection + velocity commands.
    Press 'q' to quit,  't' to toggle tuning mode.
    """
    detector = BallDetector(tuning_mode=tuning_mode)
    follower = BallFollower()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ball_tracker] Could not open camera {camera_index}")
        return

    print("[ball_tracker] Running live tracker – press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det  = detector.detect(frame)
        v, w = follower.update(det)
        out  = detector.annotate(frame, det)

        status = f"v={v:.2f} m/s  w={w:.2f} rad/s"
        state  = "FOUND" if det.found else "SEARCHING"
        cv2.putText(out, f"{state}  {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("Ball Tracker", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('t'):
            detector.tuning_mode = not detector.tuning_mode

    cap.release()
    cv2.destroyAllWindows()
