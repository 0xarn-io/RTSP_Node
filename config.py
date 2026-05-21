"""Loads and validates the application's TOML configuration.

The whole node is driven by a single TOML file (``config.toml`` by
default, or whatever ``RTSP_NODE_CONFIG`` points at). This module turns
that file into typed, immutable dataclasses so the rest of the code never
touches raw dicts, and it fails fast with a clear message when the file is
missing, malformed, or logically invalid (no cameras, duplicate ids, ...).
"""
from __future__ import annotations

import os
import tomllib  # standard library TOML parser (Python 3.11+)
from dataclasses import dataclass, field
from pathlib import Path

# Filename used when the RTSP_NODE_CONFIG environment variable is not set.
DEFAULT_CONFIG_PATH = "config.toml"
# Environment variable that overrides the config path (e.g. to point at a
# private file outside the repo on a real deployment).
CONFIG_ENV_VAR = "RTSP_NODE_CONFIG"


# frozen=True makes instances read-only (hashable, can't be mutated by
# accident after they're loaded once at startup).
@dataclass(frozen=True)
class CameraConfig:
    """One [[cameras]] entry from the TOML file."""

    id: str           # unique key used in API paths, e.g. /cameras/<id>
    url: str           # full RTSP URL the capture thread connects to
    name: str = ""     # human-friendly label shown in status responses
    # If > 0, a frame older than this many seconds is treated as missing
    # (protects against serving a stale image from a frozen stream).
    max_frame_age_s: float = 0.0
    # Optional crop box [x, y, width, height] in pixels. When set, served
    # screenshots are cropped to this region (e.g. [826, 345, 832, 832]
    # yields 832x832 images). None = serve the full frame.
    roi: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ServerConfig:
    """The [server] table: where this node's own HTTP API binds.

    Only used by the ``python main.py`` entrypoint; ignored when launched
    via the uvicorn/gunicorn CLI directly.
    """

    host: str = "0.0.0.0"  # 0.0.0.0 = listen on every network interface
    port: int = 8000


@dataclass(frozen=True)
class Config:
    """The fully parsed configuration file."""

    # default_factory is required because dataclass fields can't share a
    # single mutable default instance across objects.
    server: ServerConfig = field(default_factory=ServerConfig)
    cameras: list[CameraConfig] = field(default_factory=list)


def config_path() -> Path:
    """Resolve which file to load: the env override, else the default."""
    return Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))


def _parse_roi(cam_id: str, raw_roi) -> tuple[int, int, int, int] | None:
    """Validate an optional [x, y, width, height] crop box from the TOML."""
    if raw_roi is None:
        return None
    if not isinstance(raw_roi, (list, tuple)) or len(raw_roi) != 4:
        raise ValueError(
            f"Camera '{cam_id}' roi must be a 4-element [x, y, width, height]."
        )
    x, y, w, h = (int(v) for v in raw_roi)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(
            f"Camera '{cam_id}' roi needs x,y >= 0 and width,height > 0."
        )
    return (x, y, w, h)


def load_config(path: Path | None = None) -> Config:
    """Read and validate the TOML file, returning a :class:`Config`.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError``
    if its contents are invalid, so startup aborts loudly instead of the
    node running with a broken/empty camera list.
    """
    path = path or config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Create {DEFAULT_CONFIG_PATH} (or set {CONFIG_ENV_VAR}) "
            f"with [server] and [[cameras]] entries."
        )

    # tomllib requires the file to be opened in binary mode.
    with path.open("rb") as fh:
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
                roi=_parse_roi(cam_id, entry.get("roi")),
            )
        )

    # A node with no cameras can't do anything useful; refuse to start.
    if not cameras:
        raise ValueError("No cameras configured: add at least one [[cameras]] entry.")

    return Config(server=server, cameras=cameras)
