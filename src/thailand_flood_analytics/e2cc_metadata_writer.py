from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .data_sources import DataMode
from .event_config import EventConfig
from .flood_risk_model import RiskInputs, compute_risk_score
from .geospatial_processing import ReplayGrid, warning_centroids
from .warning_levels import WARNING_LEVELS, WarningLevel, info_for


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize(arr: np.ndarray, max_value: float) -> np.ndarray:
    return np.clip(arr / max_value, 0, 1)


def write_gray_jpeg(path: Path, values: np.ndarray, max_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (_normalize(values, max_value) * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path, format="JPEG", quality=95)


def color_to_rgb01(hex_color: str) -> list[float]:
    value = hex_color.lstrip("#")
    return [int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def circle_points(lat: float, lon: float, radius_deg: float = 0.045, segments: int = 36) -> list[list[float]]:
    pts = []
    for idx in range(segments + 1):
        angle = 2.0 * np.pi * idx / segments
        pts.append([lat + radius_deg * float(np.sin(angle)), lon + radius_deg * float(np.cos(angle))])
    return pts


def build_warning_badges(event: EventConfig, grid: ReplayGrid, mode: DataMode) -> list[dict[str, Any]]:
    centroids = warning_centroids(event)
    badges: list[dict[str, Any]] = []
    for t_idx, time in enumerate(grid.times):
        event_peak = float(np.nanmax(grid.flood_risk[t_idx]))
        for place in centroids:
            urban = float(place["urban"])
            population = float(place["population"])
            critical = float(place["critical"])
            factor = 0.75 + urban * 0.25
            rain24 = float(np.nanmax(grid.rainfall_24h[t_idx]) * factor)
            rain72 = float(np.nanmax(grid.rainfall_72h[t_idx]) * factor)
            depth = max(0.0, (event_peak - 55.0) / 35.0) * (0.7 + urban)
            river = min(1.0, event_peak / 100.0)
            score, level, drivers = compute_risk_score(
                RiskInputs(
                    rainfall_24h=rain24,
                    rainfall_72h=rain72,
                    river_level_percentile=river,
                    flood_depth=depth,
                    terrain_susceptibility=0.65,
                    population_exposure=population,
                    urban_exposure=urban,
                    critical_infrastructure_exposure=critical,
                    historical_recurrence=0.55,
                )
            )
            info = info_for(level)
            badges.append(
                {
                    "event_id": event.event_id,
                    "time": _iso(time),
                    "admin_level": "district",
                    "province": place["province"],
                    "district": place["district"],
                    "lat": place["lat"],
                    "lon": place["lon"],
                    "risk_score": score,
                    "warning_level": level.value,
                    "label_th": info.label_th,
                    "label_en": info.label_en,
                    "color": info.color,
                    "drivers": drivers,
                    "mode": mode.value,
                    "status_type": "forecast_or_replay",
                }
            )
    return badges


def write_outputs(event: EventConfig, grid: ReplayGrid, mode: DataMode, output_dir: Path) -> dict[str, Path]:
    event_dir = output_dir / event.event_id
    rainfall_dir = event_dir / "rainfall_24h"
    accum_dir = event_dir / "rainfall_72h"
    risk_dir = event_dir / "flood_risk"
    sources_rain: dict[str, str] = {}
    sources_accum: dict[str, str] = {}
    sources_risk: dict[str, str] = {}

    for idx, time in enumerate(grid.times):
        stamp = time.strftime("%Y%m%dT%H%M")
        rain_path = rainfall_dir / f"rain24h_{stamp}.jpg"
        accum_path = accum_dir / f"rain72h_{stamp}.jpg"
        risk_path = risk_dir / f"risk_{stamp}.jpg"
        write_gray_jpeg(rain_path, grid.rainfall_24h[idx], 250)
        write_gray_jpeg(accum_path, grid.rainfall_72h[idx], 500)
        write_gray_jpeg(risk_path, grid.flood_risk[idx], 100)
        sources_rain[_iso(time)] = str(rain_path.relative_to(output_dir))
        sources_accum[_iso(time)] = str(accum_path.relative_to(output_dir))
        sources_risk[_iso(time)] = str(risk_path.relative_to(output_dir))

    badges = build_warning_badges(event, grid, mode)
    badge_path = event_dir / "warning_badges.json"
    badge_path.write_text(json.dumps({"event_id": event.event_id, "badges": badges}, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_by_level: dict[WarningLevel, list[dict[str, Any]]] = {level: [] for level in WarningLevel}
    for badge in badges:
        latest_by_level[WarningLevel(badge["warning_level"])].append(badge)

    curve_features: list[dict[str, Any]] = []
    for level, level_badges in latest_by_level.items():
        if not level_badges:
            continue
        points = []
        points_per_curve = []
        seen = {}
        for badge in level_badges:
            key = (badge["province"], badge["district"])
            seen[key] = badge
        for badge in seen.values():
            cur = circle_points(float(badge["lat"]), float(badge["lon"]))
            points.extend(cur)
            points_per_curve.append(len(cur))
        curve_features.append(
            {
                "name": f"Warning Badge Areas - {level.value}",
                "type": "Curves",
                "active": True,
                "projection": "latlon",
                "color": color_to_rgb01(WARNING_LEVELS[level].color),
                "width": 8,
                "periodic": False,
                "points": points,
                "points_per_curve": points_per_curve,
                "meta": {
                    "event_id": event.event_id,
                    "warning_level": level.value,
                    "badge_source": str(badge_path.relative_to(output_dir)),
                    "data_mode": mode.value,
                },
            }
        )

    aoi = event.aoi
    options = {
        "utc_start_time": _iso(grid.times[0]),
        "utc_end_time": _iso(grid.timeline_end),
        "utc_time": _iso(grid.times[0]),
        "playback_duration": 30,
        "play": False,
    }
    common_meta = {
        "event_id": event.event_id,
        "data_mode": mode.value,
        "safety_note": "This is a research/demo decision-support visualization, not an official warning system.",
    }
    manifest = {
        "options": options,
        "features": [
            {
                "name": "Forecast Rainfall 24h",
                "type": "Image",
                "active": bool(event.layers.get("rainfall_forecast", True)),
                "projection": "latlong",
                "colormap": "turbo",
                "sources": sources_rain,
                "latlon_min": [aoi["lat_min"], aoi["lon_min"]],
                "latlon_max": [aoi["lat_max"], aoi["lon_max"]],
                "remapping": {"input_min": 0.0, "input_max": 250.0, "output_min": 0.0, "output_max": 1.0, "output_gamma": 1.0},
                "meta": common_meta | {"unit": "mm"},
            },
            {
                "name": "Accumulated Rainfall 72h",
                "type": "Image",
                "active": bool(event.layers.get("rainfall_accumulation", True)),
                "projection": "latlong",
                "colormap": "RRate11",
                "sources": sources_accum,
                "latlon_min": [aoi["lat_min"], aoi["lon_min"]],
                "latlon_max": [aoi["lat_max"], aoi["lon_max"]],
                "remapping": {"input_min": 0.0, "input_max": 500.0, "output_min": 0.0, "output_max": 1.0, "output_gamma": 1.0},
                "meta": common_meta | {"unit": "mm"},
            },
            {
                "name": "Flood Risk Heatmap",
                "type": "Image",
                "active": bool(event.layers.get("flood_risk", True)),
                "projection": "latlong",
                "colormap": "plasma",
                "sources": sources_risk,
                "latlon_min": [aoi["lat_min"], aoi["lon_min"]],
                "latlon_max": [aoi["lat_max"], aoi["lon_max"]],
                "remapping": {"input_min": 0.0, "input_max": 100.0, "output_min": 0.0, "output_max": 1.0, "output_gamma": 1.0},
                "meta": common_meta | {"unit": "risk_score_0_100"},
            },
        ]
        + curve_features,
        "thailand_flood_command_center": {
            "event_id": event.event_id,
            "display_name": event.display_name,
            "warning_badges": str(badge_path.relative_to(output_dir)),
            "legend": {level.value: info.__dict__ for level, info in WARNING_LEVELS.items()},
            "data_mode": mode.value,
            "safety_note": common_meta["safety_note"],
        },
    }

    manifest_path = output_dir / f"thailand_flood_command_center_{event.event_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "badges": badge_path, "event_dir": event_dir}
