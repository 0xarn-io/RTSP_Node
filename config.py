"""Configuration: load the TOML file into typed objects, and write tuned
values back to it.

The whole node is driven by one TOML file (``config.toml`` by default, or
whatever ``RTSP_NODE_CONFIG`` points at). Reading turns the file into frozen
dataclasses so the rest of the code never juggles raw dicts; it fails fast
with a clear message if the file is missing or invalid. Writing
(:func:`update_camera_in_file`) is used by the setup tool to persist tuned
rotate/roi/undistort values while preserving comments and formatting.
"""
from __future__ import annotations

import os
import tomllib  # standard-library TOML reader (Python 3.11+)
from dataclasses import dataclass, field
from pathlib import Path

# Config file used when RTSP_NODE_CONFIG is not set.
DEFAULT_CONFIG_PATH = "config.toml"
# Env var that overrides the config path (e.g. to point at a private file
# outside the repo on a real deployment).
CONFIG_ENV_VAR = "RTSP_NODE_CONFIG"


# frozen=True makes instances read-only (so config can't be mutated by
# accident after startup). The capture worker swaps in a *new* CameraConfig
# via dataclasses.replace when the setup tool changes a value live.
@dataclass(frozen=True)
class CameraConfig:
    """One ``[[cameras]]`` entry from the TOML file."""

    id: str            # unique key used in API paths, e.g. /cameras/<id>
    url: str           # full RTSP URL the capture thread connects to
    name: str = ""     # human-friendly label shown in status responses
    # If > 0, a frame older than this many seconds is treated as missing
    # (guards against serving a stale image from a frozen stream).
    max_frame_age_s: float = 0.0
    # Crop box [x, y, width, height] in pixels, applied last in the pipeline.
    # e.g. [826, 345, 832, 832] yields 832x832 images. None = full frame.
    roi: tuple[int, int, int, int] | None = None
    # Rotation in degrees (CCW), applied after undistort and before crop.
    # Use it to level/orient a tilted camera. 0 = no rotation.
    rotate: float = 0.0
    # Lens undistortion [k1, k2, focal_ratio], applied first in the pipeline.
    # k1/k2 are radial coefficients; focal_ratio is the focal length as a
    # fraction of frame width. None (or k1=k2=0) means no correction.
    undistort: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ServerConfig:
    """The ``[server]`` table: where this node's own HTTP API binds.

    Only used by the ``python main.py`` entrypoint; ignored when launched
    via the uvicorn/gunicorn CLI directly.
    """

    host: str = "0.0.0.0"  # 0.0.0.0 = listen on every network interface
    port: int = 8000


@dataclass(frozen=True)
class Config:
    """The fully parsed configuration file."""

    # default_factory: dataclass fields can't share one mutable default.
    server: ServerConfig = field(default_factory=ServerConfig)
    cameras: list[CameraConfig] = field(default_factory=list)


# --- parsing/validation helpers (also reused by the API to validate input) ---

def parse_roi(cam_id: str, raw) -> tuple[int, int, int, int] | None:
    """Validate an optional [x, y, width, height] crop box."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"Camera '{cam_id}' roi must be a 4-element [x, y, width, height].")
    x, y, w, h = (int(v) for v in raw)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Camera '{cam_id}' roi needs x,y >= 0 and width,height > 0.")
    return (x, y, w, h)


def parse_undistort(cam_id: str, raw) -> tuple[float, float, float] | None:
    """Validate an optional [k1, k2, focal_ratio] undistortion triple."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"Camera '{cam_id}' undistort must be [k1, k2, focal_ratio].")
    k1, k2, focal_ratio = (float(v) for v in raw)
    if focal_ratio <= 0:
        raise ValueError(f"Camera '{cam_id}' undistort focal_ratio must be > 0.")
    return (k1, k2, focal_ratio)


# --- read ---------------------------------------------------------------------

def config_path() -> Path:
    """Resolve which file to load: the env override, else the default."""
    return Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))


def load_config(path: Path | None = None) -> Config:
    """Read and validate the TOML file, returning a :class:`Config`.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError``
    if its contents are invalid, so startup aborts loudly rather than
    running with a broken/empty camera list.
    """
    path = path or config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Create {DEFAULT_CONFIG_PATH} (or set {CONFIG_ENV_VAR}) "
            f"with [server] and [[cameras]] entries."
        )

    with path.open("rb") as fh:  # tomllib requires binary mode
        raw = tomllib.load(fh)

    # [server] is optional; fall back to the dataclass defaults per field.
    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=str(server_raw.get("host", ServerConfig.host)),
        port=int(server_raw.get("port", ServerConfig.port)),
    )

    # Parse every [[cameras]] entry, validating as we go. `seen` enforces
    # unique ids because the id is the API key for each camera.
    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for entry in raw.get("cameras", []):
        cam_id = str(entry.get("id", "")).strip()
        url = str(entry.get("url", "")).strip()
        if not cam_id:
            raise ValueError("Every [[cameras]] entry needs a non-empty 'id'.")
        if not url:
            raise ValueError(f"Camera '{cam_id}' is missing a non-empty 'url'.")
        if cam_id in seen:
            raise ValueError(f"Duplicate camera id: '{cam_id}'.")
        seen.add(cam_id)
        cameras.append(
            CameraConfig(
                id=cam_id,
                url=url,
                name=str(entry.get("name", "")).strip(),
                max_frame_age_s=float(entry.get("max_frame_age_s", 0.0)),
                roi=parse_roi(cam_id, entry.get("roi")),
                rotate=float(entry.get("rotate", 0.0)),
                undistort=parse_undistort(cam_id, entry.get("undistort")),
            )
        )

    if not cameras:
        raise ValueError("No cameras configured: add at least one [[cameras]] entry.")

    return Config(server=server, cameras=cameras)


# --- write --------------------------------------------------------------------

def update_camera_in_file(
    camera_id: str,
    *,
    rotate: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    undistort: tuple[float, float, float] | None = None,
    path: Path | None = None,
) -> None:
    """Persist ``rotate``/``roi``/``undistort`` for one camera back to the
    TOML file, preserving comments and formatting.

    Only the keyword arguments that are not None are written. Uses tomlkit,
    imported lazily so the read path never depends on it. Raises KeyError if
    the camera id is not present in the file.
    """
    import tomlkit  # only needed for writing; keeps the read path dependency-free

    path = path or config_path()
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    for entry in doc.get("cameras", []):
        if str(entry.get("id", "")).strip() == camera_id:
            if rotate is not None:
                entry["rotate"] = rotate
            if roi is not None:
                entry["roi"] = list(roi)
            if undistort is not None:
                entry["undistort"] = list(undistort)
            break
    else:  # for/else: ran without `break` -> id never matched
        raise KeyError(f"Camera '{camera_id}' not found in {path}")
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")
