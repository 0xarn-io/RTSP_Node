from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = "config.toml"
CONFIG_ENV_VAR = "RTSP_NODE_CONFIG"


@dataclass(frozen=True)
class CameraConfig:
    id: str
    url: str
    name: str = ""
    # Drop frames older than this before serving (0 = serve whatever is latest).
    max_frame_age_s: float = 0.0


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    cameras: list[CameraConfig] = field(default_factory=list)


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Create {DEFAULT_CONFIG_PATH} (or set {CONFIG_ENV_VAR}) "
            f"with [server] and [[cameras]] entries."
        )

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=str(server_raw.get("host", ServerConfig.host)),
        port=int(server_raw.get("port", ServerConfig.port)),
    )

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
            )
        )

    if not cameras:
        raise ValueError("No cameras configured: add at least one [[cameras]] entry.")

    return Config(server=server, cameras=cameras)
