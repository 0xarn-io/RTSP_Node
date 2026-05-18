"""FastAPI application: the HTTP "endpoint" half of the node.

This module wires the camera capture layer (camera.py) to a small REST
API. It exposes node/camera status and an on-demand JPEG screenshot per
camera. The capture threads are started and stopped together with the
app via FastAPI's ``lifespan`` hook.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import cv2
from fastapi import Depends, FastAPI, HTTPException, Query, Response

from camera import CameraManager, CameraWorker
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Process start time, used to report uptime on /health. monotonic() is
# immune to wall-clock adjustments (NTP, DST), so the delta is reliable.
START = time.monotonic()

APP_INFO = {
    "name":            "RTSP Driver Node",
    "description":     "Restful RTSP stream driver",
    "version":         "0.0.2",
    "author":          "Amontplet",
    "email":           "amontplet@warak.com",
    "company":         "Warak Group",
    "company_website": "https://warak.com",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop the camera threads alongside the web server.

    Everything before ``yield`` runs once on startup; everything after it
    runs once on shutdown. Storing objects on ``app.state`` makes them
    reachable from request handlers via dependencies (see get_manager).
    """
    config = load_config()
    manager = CameraManager(config.cameras)
    manager.start_all()
    app.state.config = config
    app.state.manager = manager
    try:
        yield  # <-- the app serves requests while suspended here
    finally:
        # Runs even if startup/serving raised, so threads don't leak.
        manager.stop_all()


app = FastAPI(
    title=APP_INFO["name"],
    description=APP_INFO["description"],
    version=APP_INFO["version"],
    contact={
        "name":  APP_INFO["author"],
        "email": APP_INFO["email"],
        "url":   APP_INFO["company_website"],
    },
    lifespan=lifespan,
)


# --- dependencies ---------------------------------------------------------
# FastAPI "dependencies" are just functions it calls before a handler and
# injects the return value into. They keep lookup/validation out of the
# route bodies and produce proper HTTP errors automatically.

def get_manager() -> CameraManager:
    """Return the CameraManager created in ``lifespan``."""
    return app.state.manager


def get_camera(camera_id: str, manager: CameraManager = Depends(get_manager)) -> CameraWorker:
    """Resolve a path's ``{camera_id}`` to its worker, or 404 if unknown.

    Any route that declares ``Depends(get_camera)`` automatically gets the
    404 behavior without repeating it.
    """
    worker = manager.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")
    return worker


# --- routes ---------------------------------------------------------------

@app.get("/")
async def about():
    """Static node metadata (name, version, contact)."""
    return APP_INFO


@app.get("/health")
async def health(manager: CameraManager = Depends(get_manager)):
    """Liveness + a quick summary of how many cameras are connected."""
    cameras = manager.status_all()
    return {
        "status": "ok",
        "uptime_s": round(time.monotonic() - START, 1),
        "cameras_total": len(cameras),
        "cameras_connected": sum(1 for c in cameras if c["connected"]),
    }


@app.get("/cameras")
async def list_cameras(manager: CameraManager = Depends(get_manager)):
    """Live status for every configured camera."""
    return {"cameras": manager.status_all()}


@app.get("/cameras/{camera_id}")
async def camera_status(camera: CameraWorker = Depends(get_camera)):
    """Live status for a single camera (404 if the id is unknown)."""
    return camera.status()


# NOTE: this handler is a plain `def`, not `async def`, on purpose.
# cv2.imencode is blocking CPU work; declaring it sync makes FastAPI run
# it in a threadpool so it doesn't stall the async event loop.
@app.get("/cameras/{camera_id}/screenshot")
def screenshot(
    camera: CameraWorker = Depends(get_camera),
    quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
):
    """Return the camera's latest frame as a JPEG image.

    404 = unknown camera (from get_camera), 503 = connected/connecting but
    no frame captured yet, 500 = the frame failed to JPEG-encode.
    Frame metadata is returned in ``X-Frame-*`` response headers so clients
    can read it without decoding the image.
    """
    snap = camera.snapshot()
    if snap is None:
        raise HTTPException(
            status_code=503,
            detail=f"No frame available yet for camera '{camera.cam.id}'.",
        )
    frame, ts = snap
    # imencode compresses the raw BGR array into JPEG bytes in `buf`.
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame.")

    height, width = frame.shape[:2]  # numpy shape is (rows, cols, channels)
    headers = {
        "X-Camera-Id": camera.cam.id,
        "X-Frame-Timestamp": f"{ts:.3f}",  # epoch seconds the frame was grabbed
        "X-Frame-Age-Ms": str(int((time.time() - ts) * 1000)),  # staleness
        "X-Frame-Width": str(width),
        "X-Frame-Height": str(height),
        "Cache-Control": "no-store",  # a live snapshot must never be cached
    }
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers=headers)


# Allows `python main.py` to run the server using the [server] section of
# the config. (Production would more likely use the uvicorn/gunicorn CLI.)
if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
