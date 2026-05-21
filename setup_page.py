"""The on-demand ROI / rotate / undistort setup page (HTML + JS).

A single self-contained page with no build step and no background work — it
only does anything while you have it open. It fetches /cameras/{id}/frame to
show a live (undistorted + rotated) frame, lets you draw or type an ROI, and
can POST the chosen values back to /cameras/{id}/config.

The page is a plain template string with ``__PLACEHOLDER__`` tokens;
:func:`render` substitutes one camera's current settings. The JS is vanilla
(no dependencies) so the page works in any modern browser.
"""
from __future__ import annotations

import html
import json

from config import CameraConfig

_PAGE = """<!doctype html>
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

// Current lens-undistort inputs as numbers.
function lensParams() {
  return {
    k1: parseFloat(k1El.value) || 0,
    k2: parseFloat(k2El.value) || 0,
    f: parseFloat(fEl.value) || 1,
  };
}

// (Re)load the frame from the server with the current rotation + lens
// params applied, so the preview matches exactly what the server will crop.
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

// Keep an ROI inside the image bounds (natural pixels).
function clampRoi(x, y, w, h) {
  const W = img.naturalWidth, H = img.naturalHeight;
  x = Math.max(0, Math.min(Math.round(x), W - 1));
  y = Math.max(0, Math.min(Math.round(y), H - 1));
  w = Math.max(1, Math.min(Math.round(w), W - x));
  h = Math.max(1, Math.min(Math.round(h), H - y));
  return [x, y, w, h];
}

// Single source of truth: set the ROI, then sync the inputs, box and preview.
function setRoi(x, y, w, h) {
  if (!img.naturalWidth) return;  // no frame loaded yet
  roi = clampRoi(x, y, w, h);
  rx.value = roi[0]; ry.value = roi[1]; rw.value = roi[2]; rh.value = roi[3];
  drawBox(); drawPreview(); render();
}

// Position the green overlay box from the ROI (natural -> displayed px).
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

// Drag on the image to draw an ROI (displayed px during drag, converted to
// natural px on release via setRoi).
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

// Draw the cropped region into the preview canvas at native resolution.
function drawPreview() {
  const [x, y, w, h] = roi;
  preview.width = w; preview.height = h;
  preview.getContext('2d').drawImage(img, x, y, w, h, 0, 0, w, h);
}

// Render the config snippet (and output size) for the current settings.
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
// Apply to the running camera and persist to config.toml.
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


def render(cam: CameraConfig) -> str:
    """Return the setup page HTML for ``cam`` with its current settings filled in.

    Values are injected safely: the camera id is HTML-escaped where it
    appears as text and JSON-encoded where it appears in JS; the ROI and
    undistort defaults are seeded so the inputs open pre-populated.
    """
    undistort = cam.undistort or (0.0, 0.0, 1.0)
    roi_js = json.dumps(list(cam.roi) if cam.roi else None)
    # Replace the JS token before the plain one (it contains it as a prefix).
    return (
        _PAGE
        .replace("__CAMERA_ID_JS__", json.dumps(cam.id))
        .replace("__ROI_JS__", roi_js)
        .replace("__CAMERA_ID__", html.escape(cam.id))
        .replace("__ROTATE__", str(cam.rotate))
        .replace("__K1__", str(undistort[0]))
        .replace("__K2__", str(undistort[1]))
        .replace("__F__", str(undistort[2]))
    )
