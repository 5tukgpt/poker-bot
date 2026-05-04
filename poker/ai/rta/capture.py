"""Capture-card video frame ingestion for RTA pipeline.

Reads frames from an HDMI capture card (e.g., Elgato Cam Link 4K) connected
to Computer 2 via USB. Outputs frames to the OCR module.

Usage on Computer 2:
    from poker.ai.rta.capture import CaptureSource

    cap = CaptureSource(device_index=0)  # /dev/video0 on Linux, etc.
    while True:
        frame = cap.read_frame()
        if frame is None:
            continue
        # ... feed frame into OCR pipeline ...
"""

from __future__ import annotations

import time
from typing import Iterator

# OpenCV import is wrapped — only needed on Computer 2 with capture hardware
try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False
    cv2 = None
    np = None


class CaptureSource:
    """Reads frames from a capture card (or fallback to test video file)."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps_target: int = 5,  # we don't need 60fps; 5fps catches all action changes
        test_video_path: str | None = None,
    ) -> None:
        if not HAS_CV:
            raise ImportError(
                "opencv-python required. Install with: pip install opencv-python"
            )

        self.fps_target = fps_target
        self._frame_interval = 1.0 / fps_target
        self._last_read = 0.0

        if test_video_path:
            self._cap = cv2.VideoCapture(test_video_path)
        else:
            self._cap = cv2.VideoCapture(device_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open capture device {device_index}")

    def read_frame(self) -> 'np.ndarray | None':
        """Read next frame, throttled to fps_target. Returns BGR np.ndarray or None."""
        elapsed = time.time() - self._last_read
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)

        ret, frame = self._cap.read()
        self._last_read = time.time()
        if not ret:
            return None
        return frame

    def stream(self) -> Iterator['np.ndarray']:
        """Iterate over frames continuously."""
        while True:
            frame = self.read_frame()
            if frame is not None:
                yield frame

    def release(self) -> None:
        if self._cap:
            self._cap.release()


def find_table_window(frame: 'np.ndarray') -> 'tuple[int, int, int, int] | None':
    """Locate the Ignition poker table within a full-screen frame.

    Returns (x, y, width, height) of the table window, or None if not found.

    Strategy: look for the green felt color (Ignition's distinctive table color)
    using HSV color filtering, then find the largest contour.
    """
    if not HAS_CV:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Ignition table green — tune these by sampling actual screenshots
    lower_green = np.array([35, 50, 30])
    upper_green = np.array([85, 255, 200])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50_000:  # too small — probably not the table
        return None

    return cv2.boundingRect(largest)
