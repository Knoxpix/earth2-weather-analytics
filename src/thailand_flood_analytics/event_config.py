from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "thailand_events.yaml"


def parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_start_utc(value: str | datetime) -> datetime:
    return parse_utc(f"{value}T00:00:00Z" if isinstance(value, str) and len(value) == 10 else value)


def parse_end_utc(value: str | datetime) -> datetime:
    return parse_utc(f"{value}T23:59:59Z" if isinstance(value, str) and len(value) == 10 else value)


def expand_env_path(value: str) -> Path:
    env = os.environ.copy()
    env.setdefault("EARTH2STUDIO_PROJECT_DIR", "/home/siamai/earth2studio-project")
    env.setdefault("THAILAND_FLOOD_DATA_DIR", str(REPO_ROOT / "data"))
    env.setdefault("E2CC_OUTPUT_DIR", str(REPO_ROOT / "outputs" / "e2cc"))
    expanded = value
    for key, replacement in env.items():
        expanded = expanded.replace("${" + key + "}", replacement)
        expanded = expanded.replace("$" + key, replacement)
    expanded = os.path.expandvars(expanded)
    return Path(expanded).expanduser()


@dataclass(frozen=True)
class EventConfig:
    event_id: str
    display_name: str
    country: str
    aoi: dict[str, float]
    centroid: dict[str, float]
    date_range: dict[str, str]
    forecast_init_times: list[str]
    layers: dict[str, bool]
    ground_truth: dict[str, Any]
    cached_dataset_candidates: tuple[str, ...] = ()

    @property
    def start_time(self) -> datetime:
        return parse_utc(self.date_range["start"])

    @property
    def end_time(self) -> datetime:
        return parse_utc(self.date_range["end"])

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EventConfig":
        return cls(
            event_id=str(data["event_id"]),
            display_name=str(data["display_name"]),
            country=str(data.get("country", "Thailand")),
            aoi={k: float(v) for k, v in data["aoi"].items()},
            centroid={k: float(v) for k, v in data["centroid"].items()},
            date_range=dict(data["date_range"]),
            forecast_init_times=[str(v) for v in data.get("forecast_init_times", [])],
            layers={k: bool(v) for k, v in data.get("layers", {}).items()},
            ground_truth=dict(data.get("ground_truth", {})),
            cached_dataset_candidates=tuple(data.get("cached_dataset_candidates", []) or ()),
        )

    def filter_range(self, start: str | None = None, end: str | None = None) -> tuple[datetime, datetime]:
        selected_start = parse_start_utc(start) if start else self.start_time
        selected_end = parse_end_utc(end) if end else self.end_time
        if selected_start < self.start_time:
            selected_start = self.start_time
        if selected_end > self.end_time:
            selected_end = self.end_time
        if selected_start > selected_end:
            raise ValueError(f"Invalid date range for {self.event_id}: start is after end")
        return selected_start, selected_end

    def cached_paths(self) -> list[Path]:
        return [expand_env_path(p) for p in self.cached_dataset_candidates]


def load_events(path: str | Path | None = None) -> dict[str, EventConfig]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs/thailand_events.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    events = [EventConfig.from_mapping(item) for item in raw["events"]]
    return {event.event_id: event for event in events}


def get_event(event_id: str, path: str | Path | None = None) -> EventConfig:
    events = load_events(path)
    try:
        return events[event_id]
    except KeyError as exc:
        raise KeyError(f"Unknown event_id: {event_id}") from exc
