"""Pure image-processing pipeline — no capture, no HTTP, just pixels.

A frame flows through three optional stages, always in this order:

    raw frame  ->  undistort  ->  rotate  ->  crop to ROI  ->  result

Each stage is a small pure function (numpy array in, numpy array out) that
is a no-op when its parameter is unset. Keeping them here makes the
pipeline easy to read, reuse (the setup tool previews the same
undistort+rotate the server applies), and test in isolation.

Note: the stages may return a view of, or the very same, input array (e.g.
a no-op stage, or the crop's slice). Callers that hand the result outside
the capture lock or to an encoder should ``.copy()`` it — the capture
worker does exactly that.
"""
from __future__ import annotations

import cv2
import numpy as np


def undistort(frame, params: tuple[float, float, float] | None):
    """Correct lens (barrel/fisheye) distortion with a simple radial model.

    ``params`` is ``(k1, k2, focal_ratio)`` or None. k1/k2 are the radial
    distortion coefficients; the camera matrix is assumed centered with
    focal length ``focal_ratio * frame_width``. Output keeps the input
    width/height. No-op when params is None or k1 == k2 == 0.
    """
    if params is None:
        return frame
    k1, k2, focal_ratio = params
    if k1 == 0.0 and k2 == 0.0:
        return frame
    height, width = frame.shape[:2]
    focal = focal_ratio * width
    camera_matrix = np.array([[focal, 0.0, width / 2.0],
                              [0.0, focal, height / 2.0],
                              [0.0, 0.0, 1.0]])
    coeffs = np.array([k1, k2, 0.0, 0.0, 0.0])  # [k1, k2, p1, p2, k3]
    return cv2.undistort(frame, camera_matrix, coeffs)


def rotate(frame, degrees: float):
    """Rotate ``frame`` about its center by ``degrees`` (CCW positive).

    Width/height are preserved so coordinates stay in a stable space
    regardless of angle; corners exposed by the rotation are filled black.
    No-op when ``degrees`` is 0.
    """
    if not degrees:
        return frame
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(frame, matrix, (width, height))


def crop_to_roi(frame, roi: tuple[int, int, int, int] | None):
    """Crop ``frame`` to ``roi`` = (x, y, width, height), or return it whole.

    Same convention as the downstream vision.crop_to_roi: ``frame[y:y+h,
    x:x+w]``. A ROI that runs past the frame raises ValueError rather than
    being silently clipped — a wrong-sized crop would corrupt a fixed-input
    consumer such as a CNN.
    """
    if roi is None:
        return frame
    x, y, w, h = roi
    frame_h, frame_w = frame.shape[:2]
    if x + w > frame_w or y + h > frame_h:
        raise ValueError(f"ROI x={x} y={y} {w}x{h} exceeds frame {frame_w}x{frame_h}")
    return frame[y:y + h, x:x + w]


def process_frame(frame, *, undistort_params, rotate_degrees, roi):
    """Run the full pipeline (undistort -> rotate -> crop) and return it.

    May return a view of / the same array as the input; copy before use if
    needed. Raises ValueError if the ROI exceeds the (rotated) frame.
    """
    frame = undistort(frame, undistort_params)
    frame = rotate(frame, rotate_degrees)
    return crop_to_roi(frame, roi)


def encode_jpeg(frame, quality: int = 90) -> bytes:
    """Encode a BGR frame to JPEG bytes. Raises ValueError if it fails."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to JPEG-encode frame")
    return buf.tobytes()
