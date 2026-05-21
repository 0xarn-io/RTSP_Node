"""RTSP Driver Node — FastAPI application (the HTTP layer).

Thin wiring over the other modules:
  * config.py      - load/validate/persist the TOML config
  * camera.py      - background RTSP capture (one worker per camera)
  * imaging.py     - the undistort -> rotate -> crop frame pipeline + JPEG
  * setup_page.py  - the on-demand ROI/rotate/undistort tuning page

This module defines the routes, starts/stops the capture threads alongside
the app (via lifespan), and offers a ``python main.py`` entry point.
"""
from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager
from uuid import UUID

import piexif
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import imaging
import setup_page
from camera import CameraManager, CameraWorker
from config import load_config, parse_roi, parse_undistort, update_camera_in_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Process start time for /health uptime. monotonic() is immune to wall-clock
# adjustments (NTP, DST), so the delta is reliable.
START = time.monotonic()

APP_INFO = {
    "name":            "RTSP Driver Node",
    "description":     "Restful RTSP stream driver",
    "version":         "0.1.0",
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
    reachable from request handlers via the get_manager dependency.
    """
    config = load_config()
    manager = CameraManager(config.cameras)
    manager.start_all()
    app.state.config = config
    app.state.manager = manager
    try:
        yield  # the app serves requests while suspended here
    finally:
        manager.stop_all()  # runs even if serving raised, so threads don't leak


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
# FastAPI dependencies are functions it calls before a handler, injecting
# the return value. They keep lookup/validation out of the route bodies.

def get_manager() -> CameraManager:
    """Return the CameraManager created in ``lifespan``."""
    return app.state.manager


def get_camera(camera_id: str, manager: CameraManager = Depends(get_manager)) -> CameraWorker:
    """Resolve a path's ``{camera_id}`` to its worker, or raise 404."""
    worker = manager.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")
    return worker


class ConfigUpdate(BaseModel):
    """Body for POST /cameras/{id}/config (sent by the setup tool).

    Any field left out is unchanged; fields present are validated and applied.
    """
    rotate: float = 0.0
    roi: list[int] | None = None          # [x, y, width, height]
    undistort: list[float] | None = None  # [k1, k2, focal_ratio]


# --- info routes ----------------------------------------------------------

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


# --- image routes ---------------------------------------------------------
# These handlers are plain `def` (not `async def`) on purpose: JPEG encoding
# is blocking CPU work, so FastAPI runs them in a threadpool instead of
# stalling the async event loop.

@app.get("/cameras/{camera_id}/screenshot")
def screenshot(
    camera: CameraWorker = Depends(get_camera),
    quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
    uuid: str | None = Query(
        None,
        description="Optional UUID (dashed or 32-char hex) stamped into "
                    "the JPEG's EXIF ImageUniqueID tag for traceability.",
    ),
):
    """Return the camera's latest (undistorted, rotated, cropped) frame as JPEG.

    404 = unknown camera, 503 = no frame captured yet, 500 = the ROI exceeds
    the frame or encoding failed, 400 = the supplied ``uuid`` is invalid.
    Frame metadata is returned in ``X-Frame-*`` headers; when ``uuid`` is
    given it is also written into the JPEG's EXIF so the id travels with the
    bytes (handy when the screenshot is stored or relayed).
    """
    # Validate the UUID first (cheap) and normalize to canonical 32-char hex.
    image_uid_hex: str | None = None
    if uuid is not None:
        try:
            image_uid_hex = UUID(uuid).hex
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid UUID: {uuid!r}")

    try:
        snap = camera.snapshot()  # may raise ValueError if the ROI is too big
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if snap is None:
        raise HTTPException(
            status_code=503,
            detail=f"No frame available yet for camera '{camera.cam.id}'.",
        )
    frame, ts = snap
    try:
        jpeg_bytes = imaging.encode_jpeg(frame, quality)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # cv2 writes no EXIF, so splice the UUID in with piexif if requested.
    # The dict is keyed by IFD ("Exif" = the sub-IFD that holds ImageUniqueID).
    if image_uid_hex is not None:
        exif_bytes = piexif.dump({"Exif": {piexif.ExifIFD.ImageUniqueID: image_uid_hex.encode("ascii")}})
        out = io.BytesIO()
        piexif.insert(exif_bytes, jpeg_bytes, out)
        jpeg_bytes = out.getvalue()

    height, width = frame.shape[:2]  # numpy shape is (rows, cols, channels)
    headers = {
        "X-Camera-Id": camera.cam.id,
        "X-Frame-Timestamp": f"{ts:.3f}",                       # epoch seconds grabbed
        "X-Frame-Age-Ms": str(int((time.time() - ts) * 1000)),  # staleness
        "X-Frame-Width": str(width),
        "X-Frame-Height": str(height),
        "Cache-Control": "no-store",                            # a live snapshot is never cacheable
    }
    if image_uid_hex is not None:
        headers["X-Frame-UUID"] = image_uid_hex  # so clients needn't parse EXIF
    return Response(content=jpeg_bytes, media_type="image/jpeg", headers=headers)


@app.get("/cameras/{camera_id}/frame")
def raw_frame(
    camera: CameraWorker = Depends(get_camera),
    rotate: float = Query(0.0, description="Rotate the full frame this many degrees (CCW)."),
    k1: float = Query(0.0, description="Radial distortion k1 for the undistortion preview."),
    k2: float = Query(0.0, description="Radial distortion k2 for the undistortion preview."),
    f: float = Query(1.0, gt=0, description="Focal length as a fraction of frame width."),
    quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
):
    """The full, *uncropped* frame as JPEG: undistorted, then rotated.

    Backs the setup tool — you draw the ROI on exactly the image the server
    would crop. 404 = unknown camera, 503 = no frame yet, 500 = encode error.
    """
    raw = camera.latest_raw()
    if raw is None:
        raise HTTPException(
            status_code=503,
            detail=f"No frame available yet for camera '{camera.cam.id}'.",
        )
    frame, _ = raw
    frame = imaging.undistort(frame, (k1, k2, f))
    frame = imaging.rotate(frame, rotate)
    try:
        jpeg_bytes = imaging.encode_jpeg(frame, quality)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    height, width = frame.shape[:2]
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"X-Frame-Width": str(width), "X-Frame-Height": str(height), "Cache-Control": "no-store"},
    )


# --- setup tool -----------------------------------------------------------

@app.get("/cameras/{camera_id}/setup", response_class=HTMLResponse)
def setup(camera: CameraWorker = Depends(get_camera)):
    """Serve the on-demand ROI/rotate/undistort setup page for one camera."""
    # no-store: always serve the current page (avoids a stale cached copy
    # after the node is updated).
    return HTMLResponse(content=setup_page.render(camera.cam), headers={"Cache-Control": "no-store"})


@app.post("/cameras/{camera_id}/config")
def update_camera_config(body: ConfigUpdate, camera: CameraWorker = Depends(get_camera)):
    """Apply rotate/roi/undistort to a running camera and persist to config.toml.

    Powers the setup tool's "Apply" button. 422 = invalid roi/undistort,
    500 = writing config.toml failed.
    """
    try:
        roi = parse_roi(camera.cam.id, body.roi)
        undistort = parse_undistort(camera.cam.id, body.undistort)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Persist first, so a write failure leaves the running config untouched.
    try:
        update_camera_in_file(camera.cam.id, rotate=body.rotate, roi=roi, undistort=undistort)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {exc}")
    camera.update_config(rotate=body.rotate, roi=roi, undistort=undistort)
    return camera.status()


# Allows `python main.py` to run the server using the [server] section of the
# config. (Production would more likely use the uvicorn/gunicorn CLI.)
if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
