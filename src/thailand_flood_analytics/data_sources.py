from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path

from .event_config import EventConfig


class DataMode(str, Enum):
    REAL_EARTH2STUDIO_INFERENCE = "REAL_EARTH2STUDIO_INFERENCE"
    REAL_CACHED_DATASET = "REAL_CACHED_DATASET"
    REAL_OBSERVATION_ONLY = "REAL_OBSERVATION_ONLY"
    SYNTHETIC_DEMO_FALLBACK = "SYNTHETIC_DEMO_FALLBACK"


def default_earth2studio_project_dir() -> Path:
    return Path(os.environ.get("EARTH2STUDIO_PROJECT_DIR", "/home/siamai/earth2studio-project")).expanduser()


def resolve_data_mode(event: EventConfig, requested_mode: str = "auto") -> tuple[DataMode, dict[str, str]]:
    mode = requested_mode.upper()
    details: dict[str, str] = {}

    if mode in DataMode.__members__:
        return DataMode[mode], details
    if requested_mode in {m.value for m in DataMode}:
        return DataMode(requested_mode), details
    if requested_mode.lower() != "auto":
        raise ValueError(f"Unsupported data mode: {requested_mode}")

    for path in event.cached_paths():
        if path.exists():
            details["cached_manifest"] = str(path)
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                if meta.get("mode") == DataMode.REAL_EARTH2STUDIO_INFERENCE.value:
                    return DataMode.REAL_EARTH2STUDIO_INFERENCE, details
            except Exception:
                pass
            return DataMode.REAL_CACHED_DATASET, details

    return DataMode.SYNTHETIC_DEMO_FALLBACK, {
        "fallback_reason": "No configured real Earth2Studio, cached, or observation dataset was found."
    }
