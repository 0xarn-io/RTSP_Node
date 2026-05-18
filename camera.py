from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import cv2

from config import CameraConfig

logger = logging.getLogger("rtsp_node.camera")

# Force RTSP over TCP: UDP loses packets and produces torn/green frames.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

_RECONNECT_BACKOFF_START_S = 1.0
_RECONNECT_BACKOFF_MAX_S = 30.0
# Consecutive failed reads before we treat the stream as dead and reconnect.
_MAX_READ_FAILURES = 30


class CameraWorker:
    """Owns one RTSP connection in a background thread, always holding the
    latest decoded frame so screenshots are served without per-request
    connection latency."""

    def __init__(self, cam: CameraConfig):
        self.cam = cam
        self._lock = threading.Lock()
        self._frame = None  # numpy.ndarray (BGR) or None
        self._frame_ts: float = 0.0
        self._connected = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.cam.id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # --- capture loop ------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.cam.url, cv2.CAP_FFMPEG)
        # Keep the internal buffer tiny so we read frames close to real time.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        return cap

    def _run(self) -> None:
        backoff = _RECONNECT_BACKOFF_START_S
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                cap.release()
                self._set_disconnected()
                logger.warning(
                    "camera %s: cannot open stream, retrying in %.0fs",
                    self.cam.id,
                    backoff,
                )
                self._stop.wait(backoff)
                backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX_S)
                continue

            logger.info("camera %s: connected", self.cam.id)
            backoff = _RECONNECT_BACKOFF_START_S
            self._connected = True
            failures = 0

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= _MAX_READ_FAILURES:
                        logger.warning(
                            "camera %s: stream stalled, reconnecting", self.cam.id
                        )
                        break
                    self._stop.wait(0.05)
                    continue

                failures = 0
                with self._lock:
                    self._frame = frame
                    self._frame_ts = time.time()

            cap.release()
            self._set_disconnected()

        self._set_disconnected()
        logger.info("camera %s: worker stopped", self.cam.id)

    def _set_disconnected(self) -> None:
        self._connected = False

    # --- readers -----------------------------------------------------------

    def snapshot(self):
        """Return (frame_copy, timestamp) of the latest frame, or None.

        Honors max_frame_age_s: a frame older than the configured limit is
        treated as unavailable (stale stream)."""
        with self._lock:
            if self._frame is None:
                return None
            age = time.time() - self._frame_ts
            if self.cam.max_frame_age_s and age > self.cam.max_frame_age_s:
                return None
            return self._frame.copy(), self._frame_ts

    def status(self) -> dict:
        with self._lock:
            has_frame = self._frame is not None
            ts = self._frame_ts
            shape = self._frame.shape if has_frame else None
        info = {
            "id": self.cam.id,
            "name": self.cam.name,
            "connected": self._connected,
            "has_frame": has_frame,
        }
        if has_frame:
            info["last_frame_ts"] = round(ts, 3)
            info["last_frame_age_s"] = round(time.time() - ts, 3)
            info["resolution"] = {"width": int(shape[1]), "height": int(shape[0])}
        return info


class CameraManager:
    def __init__(self, cameras: list[CameraConfig]):
        self._workers: dict[str, CameraWorker] = {
            cam.id: CameraWorker(cam) for cam in cameras
        }

    def start_all(self) -> None:
        for worker in self._workers.values():
            worker.start()

    def stop_all(self) -> None:
        for worker in self._workers.values():
            worker.stop()

    def get(self, camera_id: str) -> Optional[CameraWorker]:
        return self._workers.get(camera_id)

    def status_all(self) -> list[dict]:
        return [w.status() for w in self._workers.values()]
