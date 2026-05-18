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
    config = load_config()
    manager = CameraManager(config.cameras)
    manager.start_all()
    app.state.config = config
    app.state.manager = manager
    try:
        yield
    finally:
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


def get_manager() -> CameraManager:
    return app.state.manager


def get_camera(camera_id: str, manager: CameraManager = Depends(get_manager)) -> CameraWorker:
    worker = manager.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")
    return worker


@app.get("/")
async def about():
    return APP_INFO


@app.get("/health")
async def health(manager: CameraManager = Depends(get_manager)):
    cameras = manager.status_all()
    return {
        "status": "ok",
        "uptime_s": round(time.monotonic() - START, 1),
        "cameras_total": len(cameras),
        "cameras_connected": sum(1 for c in cameras if c["connected"]),
    }


@app.get("/cameras")
async def list_cameras(manager: CameraManager = Depends(get_manager)):
    return {"cameras": manager.status_all()}


@app.get("/cameras/{camera_id}")
async def camera_status(camera: CameraWorker = Depends(get_camera)):
    return camera.status()


@app.get("/cameras/{camera_id}/screenshot")
def screenshot(
    camera: CameraWorker = Depends(get_camera),
    quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
):
    snap = camera.snapshot()
    if snap is None:
        raise HTTPException(
            status_code=503,
            detail=f"No frame available yet for camera '{camera.cam.id}'.",
        )
    frame, ts = snap
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame.")

    height, width = frame.shape[:2]
    headers = {
        "X-Camera-Id": camera.cam.id,
        "X-Frame-Timestamp": f"{ts:.3f}",
        "X-Frame-Age-Ms": str(int((time.time() - ts) * 1000)),
        "X-Frame-Width": str(width),
        "X-Frame-Height": str(height),
        "Cache-Control": "no-store",
    }
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers=headers)


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
