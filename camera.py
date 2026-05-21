"""Background RTSP capture.

Opening an RTSP stream is slow (the handshake can take 1-3 seconds), so we
never connect per request. Each camera gets one long-lived background
thread (:class:`CameraWorker`) that connects once and continuously keeps
the latest decoded frame in memory; a request just grabs that frame and
runs it through the imaging pipeline, so screenshots are near-instant.
:class:`CameraManager` owns one worker per configured camera and is
started/stopped with the app.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import replace
from typing import Optional

import cv2

import imaging
from config import CameraConfig

logger = logging.getLogger("rtsp_node.camera")

# Force RTSP over TCP. The default (UDP) silently drops packets on a busy
# network, producing torn or green/garbled frames. OpenCV's FFmpeg backend
# reads this env var when a VideoCapture is created.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# Reconnect strategy: after a failed connection wait 1s, then double the
# wait each retry up to a 30s ceiling — a down camera doesn't spin the CPU,
# a briefly-flaky one still recovers quickly.
_RECONNECT_BACKOFF_START_S = 1.0
_RECONNECT_BACKOFF_MAX_S = 30.0
# Consecutive failed reads tolerated before we decide the stream is dead
# and tear the connection down to reconnect.
_MAX_READ_FAILURES = 30


class CameraWorker:
    """Owns one RTSP connection in a background thread, always holding the
    latest decoded frame.

    Threading note: ``_frame``/``_frame_ts`` are written by the capture
    thread and read by HTTP request threads, so every access goes through
    ``_lock``. ``_stop`` is an Event used to ask the thread to exit.
    """

    def __init__(self, cam: CameraConfig):
        self.cam = cam
        self._lock = threading.Lock()
        self._frame = None             # latest frame: numpy.ndarray (BGR) or None
        self._frame_ts: float = 0.0    # wall-clock time the frame was grabbed
        self._connected = False        # True while the RTSP stream is open
        self._stop = threading.Event()  # set() asks the capture loop to stop
        self._thread: Optional[threading.Thread] = None

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn the background capture thread (idempotent)."""
        if self._thread is not None:
            return
        # daemon=True so the thread can't keep the process alive on exit.
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.cam.id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the capture loop to exit and wait for the thread to end."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # --- capture loop ------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        """Open the RTSP stream. CAP_FFMPEG forces the FFmpeg backend."""
        cap = cv2.VideoCapture(self.cam.url, cv2.CAP_FFMPEG)
        # Shrink the internal buffer to 1 frame so cap.read() returns the
        # newest frame instead of draining a backlog (low latency). Not all
        # backends honor this, hence the guarded set().
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        return cap

    def _run(self) -> None:
        """The background thread body: connect, read frames, reconnect.

        Outer loop = (re)connection with exponential backoff.
        Inner loop = read frames as fast as the stream delivers them.
        Both loops exit promptly when ``stop()`` sets ``_stop``.
        """
        backoff = _RECONNECT_BACKOFF_START_S
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                # Couldn't connect. Release, wait (interruptibly), back off.
                cap.release()
                self._connected = False
                logger.warning(
                    "camera %s: cannot open stream, retrying in %.0fs",
                    self.cam.id, backoff,
                )
                self._stop.wait(backoff)
                backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX_S)
                continue

            logger.info("camera %s: connected", self.cam.id)
            backoff = _RECONNECT_BACKOFF_START_S  # reset for the next disconnect
            self._connected = True
            failures = 0

            while not self._stop.is_set():
                # cap.read() blocks until the next frame, so this loop is
                # naturally paced by the camera's frame rate.
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= _MAX_READ_FAILURES:
                        logger.warning("camera %s: stream stalled, reconnecting", self.cam.id)
                        break  # leave inner loop -> reconnect
                    self._stop.wait(0.05)  # brief, interruptible pause
                    continue

                # Good frame: publish it as "the latest" under the lock.
                failures = 0
                with self._lock:
                    self._frame = frame
                    self._frame_ts = time.time()

            cap.release()
            self._connected = False

        logger.info("camera %s: worker stopped", self.cam.id)

    # --- reads / live updates ----------------------------------------------

    def snapshot(self):
        """Return ``(frame_copy, timestamp)`` of the latest *processed* frame,
        or None if no frame has arrived yet.

        The frame is run through the imaging pipeline (undistort -> rotate ->
        crop) using this camera's config; raises ValueError if the ROI
        exceeds the frame. Honors ``max_frame_age_s`` (a too-old frame counts
        as unavailable). Returns a contiguous copy so the caller can encode
        it without holding the lock or racing the capture thread.
        """
        with self._lock:
            if self._frame is None:
                return None
            age = time.time() - self._frame_ts
            if self.cam.max_frame_age_s and age > self.cam.max_frame_age_s:
                return None
            frame = self._frame
            ts = self._frame_ts

        # The capture thread only ever rebinds self._frame to a new array
        # (never mutates one in place), so transforming this grabbed
        # reference after releasing the lock is safe. The final .copy()
        # de-aliases from self._frame and gives cv2 a contiguous buffer.
        processed = imaging.process_frame(
            frame,
            undistort_params=self.cam.undistort,
            rotate_degrees=self.cam.rotate,
            roi=self.cam.roi,
        )
        return processed.copy(), ts

    def latest_raw(self):
        """Return ``(frame_copy, timestamp)`` of the latest *raw* frame
        (uncropped, unrotated, no undistort), or None.

        Used by the setup tool, which applies its own trial transforms and
        lets you draw the ROI on the result.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy(), self._frame_ts

    def update_config(self, *, rotate: float | None = None,
                      roi: tuple[int, int, int, int] | None = None,
                      undistort: tuple[float, float, float] | None = None) -> None:
        """Live-swap this worker's rotate/roi/undistort by replacing the
        frozen CameraConfig. snapshot() picks it up on its next call, since
        the whole self.cam object is reassigned in one atomic step. Only the
        keyword arguments that are not None are changed.
        """
        changes = {}
        if rotate is not None:
            changes["rotate"] = rotate
        if roi is not None:
            changes["roi"] = roi
        if undistort is not None:
            changes["undistort"] = undistort
        if changes:
            self.cam = replace(self.cam, **changes)

    def status(self) -> dict:
        """Snapshot of this camera's state for the status/health endpoints."""
        # Read shared state once under the lock, then build the dict outside
        # it to keep the critical section as short as possible.
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
            # frame.shape is (height, width, channels) for a color image.
            info["resolution"] = {"width": int(shape[1]), "height": int(shape[0])}
        if self.cam.rotate:
            info["rotate"] = self.cam.rotate
        if self.cam.undistort is not None and (self.cam.undistort[0] or self.cam.undistort[1]):
            k1, k2, focal_ratio = self.cam.undistort
            info["undistort"] = {"k1": k1, "k2": k2, "f": focal_ratio}
        if self.cam.roi is not None:
            x, y, w, h = self.cam.roi
            info["roi"] = {"x": x, "y": y, "width": w, "height": h}
            if has_frame:
                # The size actually served, after clipping the ROI to the
                # frame — lets a client confirm it gets e.g. 832x832.
                info["output_resolution"] = {
                    "width": max(0, min(x + w, int(shape[1])) - x),
                    "height": max(0, min(y + h, int(shape[0])) - y),
                }
        return info


class CameraManager:
    """Owns one :class:`CameraWorker` per configured camera and exposes them
    by id. Created once at app startup, torn down at shutdown."""

    def __init__(self, cameras: list[CameraConfig]):
        self._workers: dict[str, CameraWorker] = {
            cam.id: CameraWorker(cam) for cam in cameras
        }

    def start_all(self) -> None:
        """Start every camera's background capture thread."""
        for worker in self._workers.values():
            worker.start()

    def stop_all(self) -> None:
        """Stop every capture thread (called on app shutdown)."""
        for worker in self._workers.values():
            worker.stop()

    def get(self, camera_id: str) -> Optional[CameraWorker]:
        """Look up a worker by id, or None if the id is unknown."""
        return self._workers.get(camera_id)

    def status_all(self) -> list[dict]:
        """Status dicts for every camera (used by /cameras and /health)."""
        return [w.status() for w in self._workers.values()]
