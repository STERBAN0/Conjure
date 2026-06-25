"""Threaded webcam capture.

Reading frames on a background thread keeps the main loop free for
gesture inference + rendering. We always hand back the *latest* frame,
dropping stale frames intentionally to minimise input latency.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

import config


class Camera:
    def __init__(self, index: int = config.CAM_INDEX) -> None:
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DSHOW is faster on Windows
        if not self._cap.isOpened():
            # Fall back to default backend (e.g. on macOS / Linux)
            self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_REQUEST_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_REQUEST_H)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAM_REQUEST_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if config.MIRROR_INPUT:
                # Selfie-view feels natural; gesture engine assumes this.
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()
