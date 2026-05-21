"""FastAPI application: the HTTP "endpoint" half of the node.

This module wires the camera capture layer (camera.py) to a small REST
API. It exposes node/camera status and an on-demand JPEG screenshot per
camera. The capture threads are started and stopped together with the
app via FastAPI's ``lifespan`` hook.
"""
from __future__ import annotations

import html
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import UUID

import cv2
import piexif
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from camera import CameraManager, CameraWorker, rotate_image, undistort_image
from config import _parse_roi, _parse_undistort, load_config, update_camera_in_file

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
    uuid: str | None = Query(
        None,
        description="Optional UUID (dashed or 32-char hex) stamped into "
                    "the JPEG's EXIF ImageUniqueID tag for traceability.",
    ),
):
    """Return the camera's latest frame as a JPEG image.

    404 = unknown camera (from get_camera), 503 = connected/connecting but
    no frame captured yet, 500 = the frame failed to JPEG-encode or the
    configured ROI exceeds the frame, 400 = the supplied ``uuid`` is not a
    valid hex UUID.
    Frame metadata is returned in ``X-Frame-*`` response headers so clients
    can read it without decoding the image. If ``uuid`` is provided it is
    also written into the JPEG's EXIF so the identifier travels with the
    image bytes (useful when the screenshot is stored or relayed).
    """
    # Validate the UUID up front so we return 400 before doing capture
    # work, and normalize to a canonical 32-char lowercase hex string
    # (the form ImageUniqueID is specified to hold).
    image_uid_hex: str | None = None
    if uuid is not None:
        try:
            image_uid_hex = UUID(uuid).hex
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid UUID: {uuid!r}")

    try:
        snap = camera.snapshot()
    except ValueError as exc:
        # The configured ROI doesn't fit the camera's frame (resolution
        # mismatch) — surface it instead of serving a wrong-sized image.
        raise HTTPException(status_code=500, detail=str(exc))
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
    jpeg_bytes = buf.tobytes()

    # cv2.imencode writes no EXIF, so if the caller asked for the UUID to
    # be stamped, splice an EXIF block in with piexif. The EXIF dict is
    # nested by IFD ("Exif" = the EXIF sub-IFD where ImageUniqueID lives).
    if image_uid_hex is not None:
        exif_dict = {
            "Exif": {
                piexif.ExifIFD.ImageUniqueID: image_uid_hex.encode("ascii"),
            }
        }
        exif_bytes = piexif.dump(exif_dict)
        out = io.BytesIO()
        piexif.insert(exif_bytes, jpeg_bytes, out)
        jpeg_bytes = out.getvalue()

    height, width = frame.shape[:2]  # numpy shape is (rows, cols, channels)
    headers = {
        "X-Camera-Id": camera.cam.id,
        "X-Frame-Timestamp": f"{ts:.3f}",  # epoch seconds the frame was grabbed
        "X-Frame-Age-Ms": str(int((time.time() - ts) * 1000)),  # staleness
        "X-Frame-Width": str(width),
        "X-Frame-Height": str(height),
        "Cache-Control": "no-store",  # a live snapshot must never be cached
    }
    # Echo the canonical UUID so clients don't have to crack open the EXIF.
    if image_uid_hex is not None:
        headers["X-Frame-UUID"] = image_uid_hex
    return Response(content=jpeg_bytes, media_type="image/jpeg", headers=headers)


@app.get("/cameras/{camera_id}/frame")
def raw_frame(
    camera: CameraWorker = Depends(get_camera),
    rotate: float = Query(0.0, description="Rotate the full frame this many degrees (CCW) before encoding."),
    k1: float = Query(0.0, description="Radial distortion k1 for the undistortion preview."),
    k2: float = Query(0.0, description="Radial distortion k2 for the undistortion preview."),
    f: float = Query(1.0, gt=0, description="Focal length as a fraction of frame width."),
    quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
):
    """The full, *uncropped* frame as JPEG: undistorted, then rotated.

    Backs the setup tool: you draw the ROI on exactly the image the server
    would crop. 404 = unknown camera, 503 = no frame yet, 500 = encode error.
    """
    raw = camera.latest_raw()
    if raw is None:
        raise HTTPException(
            status_code=503,
            detail=f"No frame available yet for camera '{camera.cam.id}'.",
        )
    frame, _ = raw
    frame = undistort_image(frame, (k1, k2, f))
    frame = rotate_image(frame, rotate)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame.")
    height, width = frame.shape[:2]
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Frame-Width": str(width),
            "X-Frame-Height": str(height),
            "Cache-Control": "no-store",
        },
    )


# Self-contained setup page. It does nothing until you open it (no polling,
# no background work) — it just fetches /frame on demand, lets you drag an
# ROI box and pick a rotation, then prints the config lines to paste.
_SETUP_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROI + Rotate setup - __CAMERA_ID__</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1rem; background:#111; color:#eee; }
  h1 { font-size: 1.1rem; }
  .row { margin:.5rem 0; }
  .cols { display:flex; gap:1rem; flex-wrap:wrap; align-items:flex-start; }
  #stage { position:relative; display:inline-block; border:1px solid #444; cursor:crosshair; touch-action:none; }
  #frame { display:block; max-width:90vw; max-height:70vh; }
  #box { position:absolute; border:2px solid #2ecc71; background:rgba(46,204,113,.15); pointer-events:none; display:none; }
  button, input { font-size:1rem; padding:.25rem .5rem; }
  input[type=number]{ width:6rem; }
  pre { background:#000; padding:.75rem; border:1px solid #333; white-space:pre-wrap; }
  canvas { border:1px solid #444; background:#000; max-width:45vw; }
  .muted { color:#999; font-size:.85rem; }
</style>
</head>
<body>
<h1>ROI + Rotate setup &mdash; camera &quot;__CAMERA_ID__&quot;</h1>
<div class="row">
  Rotation (deg, CCW):
  <input type="number" id="angle" step="0.5" value="__ROTATE__">
  <button id="apply">Apply / refresh frame</button>
</div>
<div class="row">
  Lens undistort:
  k1 <input type="number" id="k1" step="0.01" value="__K1__">
  k2 <input type="number" id="k2" step="0.01" value="__K2__">
  f <input type="number" id="f" step="0.05" min="0.05" value="__F__">
  <span class="muted">tweak, then &ldquo;Apply / refresh frame&rdquo; until straight lines look straight.</span>
</div>
<div class="row">
  ROI:
  x <input type="number" id="rx" min="0">
  y <input type="number" id="ry" min="0">
  w <input type="number" id="rw" min="1">
  h <input type="number" id="rh" min="1">
  <button id="applyroi">Set ROI</button>
  <span class="muted">&hellip; or drag a box on the image.</span>
</div>
<div class="cols">
  <div>
    <div id="stage"><img id="frame" alt="camera frame"><div id="box"></div></div>
    <div class="row muted" id="frameinfo"></div>
  </div>
  <div>
    <div class="row">Cropped preview (<span id="outsize">none</span>):</div>
    <canvas id="preview" width="200" height="200"></canvas>
    <div class="row">Paste into this camera's <code>config.toml</code> entry:</div>
    <pre id="out">rotate = __ROTATE__</pre>
    <div class="row">
      <button id="applycfg">Apply to camera &amp; save</button>
      <span class="muted" id="applymsg"></span>
    </div>
  </div>
</div>
<script>
const camId = __CAMERA_ID_JS__;
const img = document.getElementById('frame');
const stage = document.getElementById('stage');
const box = document.getElementById('box');
const angleEl = document.getElementById('angle');
const out = document.getElementById('out');
const outsize = document.getElementById('outsize');
const frameinfo = document.getElementById('frameinfo');
const preview = document.getElementById('preview');
const rx = document.getElementById('rx');
const ry = document.getElementById('ry');
const rw = document.getElementById('rw');
const rh = document.getElementById('rh');
const applymsg = document.getElementById('applymsg');
const k1El = document.getElementById('k1');
const k2El = document.getElementById('k2');
const fEl = document.getElementById('f');
const initialRoi = __ROI_JS__;  // [x, y, w, h] from config, or null
let roi = null;  // [x, y, w, h] in natural pixels

function lensParams() {
  return {
    k1: parseFloat(k1El.value) || 0,
    k2: parseFloat(k2El.value) || 0,
    f: parseFloat(fEl.value) || 1,
  };
}

function loadFrame() {
  const a = parseFloat(angleEl.value) || 0;
  const p = lensParams();
  img.onload = () => {
    frameinfo.textContent = `frame ${img.naturalWidth}x${img.naturalHeight} (rotated ${a} deg)`;
    if (!roi && initialRoi) setRoi(initialRoi[0], initialRoi[1], initialRoi[2], initialRoi[3]);
    else if (roi) { drawBox(); drawPreview(); }
    render();
  };
  img.onerror = () => { frameinfo.textContent = 'no frame yet - is the camera connected?'; };
  img.src = `/cameras/${encodeURIComponent(camId)}/frame?rotate=${a}&k1=${p.k1}&k2=${p.k2}&f=${p.f}&_=${Date.now()}`;
}

function clampRoi(x, y, w, h) {
  const W = img.naturalWidth, H = img.naturalHeight;
  x = Math.max(0, Math.min(Math.round(x), W - 1));
  y = Math.max(0, Math.min(Math.round(y), H - 1));
  w = Math.max(1, Math.min(Math.round(w), W - x));
  h = Math.max(1, Math.min(Math.round(h), H - y));
  return [x, y, w, h];
}

function setRoi(x, y, w, h) {
  if (!img.naturalWidth) return;  // no frame loaded yet
  roi = clampRoi(x, y, w, h);
  rx.value = roi[0]; ry.value = roi[1]; rw.value = roi[2]; rh.value = roi[3];
  drawBox(); drawPreview(); render();
}

function drawBox() {
  if (!roi) { box.style.display = 'none'; return; }
  const sx = img.clientWidth / img.naturalWidth;
  const sy = img.clientHeight / img.naturalHeight;
  box.style.display = 'block';
  box.style.left = (roi[0] * sx) + 'px';
  box.style.top = (roi[1] * sy) + 'px';
  box.style.width = (roi[2] * sx) + 'px';
  box.style.height = (roi[3] * sy) + 'px';
}

let dragging = false, sx = 0, sy = 0;
stage.addEventListener('pointerdown', (e) => {
  const r = img.getBoundingClientRect();
  sx = e.clientX - r.left; sy = e.clientY - r.top;
  dragging = true;
  box.style.display = 'block';
  box.style.left = sx + 'px'; box.style.top = sy + 'px';
  box.style.width = '0px'; box.style.height = '0px';
  stage.setPointerCapture(e.pointerId);
});
stage.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const r = img.getBoundingClientRect();
  const cx = Math.max(0, Math.min(e.clientX - r.left, img.clientWidth));
  const cy = Math.max(0, Math.min(e.clientY - r.top, img.clientHeight));
  box.style.left = Math.min(sx, cx) + 'px';
  box.style.top = Math.min(sy, cy) + 'px';
  box.style.width = Math.abs(cx - sx) + 'px';
  box.style.height = Math.abs(cy - sy) + 'px';
});
stage.addEventListener('pointerup', () => {
  if (!dragging) return;
  dragging = false;
  const fx = img.naturalWidth / img.clientWidth;
  const fy = img.naturalHeight / img.clientHeight;
  setRoi(
    parseFloat(box.style.left) * fx,
    parseFloat(box.style.top) * fy,
    parseFloat(box.style.width) * fx,
    parseFloat(box.style.height) * fy
  );
});

function drawPreview() {
  const [x, y, w, h] = roi;
  preview.width = w; preview.height = h;
  preview.getContext('2d').drawImage(img, x, y, w, h, 0, 0, w, h);
}

function render() {
  const a = parseFloat(angleEl.value) || 0;
  const p = lensParams();
  let text = `rotate = ${a}`;
  if (p.k1 || p.k2) text += `\\nundistort = [${p.k1}, ${p.k2}, ${p.f}]`;
  if (roi) {
    text += `\\nroi = [${roi[0]}, ${roi[1]}, ${roi[2]}, ${roi[3]}]`;
    outsize.textContent = `${roi[2]}x${roi[3]}`;
  } else {
    outsize.textContent = 'none';
  }
  out.textContent = text;
}

document.getElementById('apply').addEventListener('click', loadFrame);
document.getElementById('applyroi').addEventListener('click', () => {
  setRoi(parseInt(rx.value) || 0, parseInt(ry.value) || 0,
         parseInt(rw.value) || 1, parseInt(rh.value) || 1);
});
document.getElementById('applycfg').addEventListener('click', async () => {
  const p = lensParams();
  const payload = { rotate: parseFloat(angleEl.value) || 0, undistort: [p.k1, p.k2, p.f] };
  if (roi) payload.roi = roi;
  applymsg.textContent = 'applying...';
  try {
    const resp = await fetch(`/cameras/${encodeURIComponent(camId)}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    applymsg.textContent = resp.ok
      ? 'applied & saved to config.toml'
      : ('error: ' + (data.detail || resp.status));
  } catch (e) {
    applymsg.textContent = 'error: ' + e;
  }
});
loadFrame();
</script>
</body>
</html>"""


@app.get("/cameras/{camera_id}/setup", response_class=HTMLResponse)
def setup_tool(camera: CameraWorker = Depends(get_camera)):
    """Serve the on-demand ROI + rotate setup page for one camera."""
    roi_js = json.dumps(list(camera.cam.roi) if camera.cam.roi else None)
    und = camera.cam.undistort or (0.0, 0.0, 1.0)
    page = (
        _SETUP_PAGE
        .replace("__CAMERA_ID_JS__", json.dumps(camera.cam.id))
        .replace("__ROI_JS__", roi_js)
        .replace("__CAMERA_ID__", html.escape(camera.cam.id))
        .replace("__ROTATE__", str(camera.cam.rotate))
        .replace("__K1__", str(und[0]))
        .replace("__K2__", str(und[1]))
        .replace("__F__", str(und[2]))
    )
    return HTMLResponse(content=page)


class ConfigUpdate(BaseModel):
    """Body for POST /cameras/{id}/config (sent by the setup tool)."""
    rotate: float = 0.0
    roi: list[int] | None = None  # [x, y, width, height]; omit to leave unchanged
    undistort: list[float] | None = None  # [k1, k2, focal_ratio]; omit to leave unchanged


@app.post("/cameras/{camera_id}/config")
def update_camera_config(body: ConfigUpdate, camera: CameraWorker = Depends(get_camera)):
    """Apply rotate/roi/undistort to a running camera and persist to config.toml.

    Powers the setup tool's "Apply" button. 422 = invalid roi/undistort,
    500 = the live update succeeded conceptually but writing config.toml failed.
    """
    try:
        roi = _parse_roi(camera.cam.id, body.roi)
        undistort = _parse_undistort(camera.cam.id, body.undistort)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Persist first, so a write failure leaves the running config untouched.
    try:
        update_camera_in_file(camera.cam.id, rotate=body.rotate, roi=roi, undistort=undistort)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {exc}")
    camera.update_config(rotate=body.rotate, roi=roi, undistort=undistort)
    return camera.status()


# Allows `python main.py` to run the server using the [server] section of
# the config. (Production would more likely use the uvicorn/gunicorn CLI.)
if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
