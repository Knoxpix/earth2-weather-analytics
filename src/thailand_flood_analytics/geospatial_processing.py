from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .event_config import EventConfig


@dataclass(frozen=True)
class ReplayGrid:
    lat: np.ndarray
    lon: np.ndarray
    times: list[datetime]
    rainfall_24h: np.ndarray
    rainfall_72h: np.ndarray
    flood_risk: np.ndarray
    timeline_end: datetime


def daily_times(start: datetime, end: datetime) -> list[datetime]:
    cur = start.replace(hour=0, minute=0, second=0, microsecond=0)
    out: list[datetime] = []
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def synthetic_event_grid(event: EventConfig, start: datetime, end: datetime, size: int = 96) -> ReplayGrid:
    aoi = event.aoi
    lat = np.linspace(aoi["lat_max"], aoi["lat_min"], size)
    lon = np.linspace(aoi["lon_min"], aoi["lon_max"], size)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    times = daily_times(start, end)

    center_lat = event.centroid["lat"]
    center_lon = event.centroid["lon"]
    spatial = np.exp(-(((yy - center_lat) / max(0.18, (aoi["lat_max"] - aoi["lat_min"]) / 5)) ** 2 + ((xx - center_lon) / max(0.18, (aoi["lon_max"] - aoi["lon_min"]) / 5)) ** 2))

    event_len = max(1, len(times) - 1)
    rain24 = []
    rain72 = []
    risk = []
    rolling: list[np.ndarray] = []
    for idx, _time in enumerate(times):
        phase = idx / event_len
        peak = np.exp(-0.5 * ((phase - 0.62) / 0.18) ** 2)
        antecedent = 0.35 * np.exp(-0.5 * ((phase - 0.35) / 0.25) ** 2)
        daily = 18 + 220 * peak * spatial + 70 * antecedent * spatial + 10 * np.sin((xx + yy + idx) * 2.3)
        daily = np.clip(daily, 0, 320)
        rolling.append(daily)
        rain24.append(daily)
        rain72.append(np.sum(rolling[-3:], axis=0))
        risk.append(np.clip((daily / 250 * 45) + (rain72[-1] / 500 * 35) + spatial * 20, 0, 100))

    return ReplayGrid(
        lat=lat,
        lon=lon,
        times=times,
        rainfall_24h=np.array(rain24),
        rainfall_72h=np.array(rain72),
        flood_risk=np.array(risk),
        timeline_end=end,
    )


def warning_centroids(event: EventConfig) -> list[dict[str, object]]:
    lat = event.centroid["lat"]
    lon = event.centroid["lon"]
    if event.event_id == "hatyai_flood_2025":
        return [
            {"province": "Songkhla", "district": "Hat Yai", "lat": 7.008, "lon": 100.474, "urban": 0.95, "population": 0.9, "critical": 0.8},
            {"province": "Songkhla", "district": "Khlong Hoi Khong", "lat": 6.90, "lon": 100.38, "urban": 0.45, "population": 0.45, "critical": 0.2},
            {"province": "Songkhla", "district": "Sadao", "lat": 6.64, "lon": 100.42, "urban": 0.55, "population": 0.55, "critical": 0.3},
        ]
    if event.event_id == "maesai_flood_2024":
        return [
            {"province": "Chiang Rai", "district": "Mae Sai", "lat": lat, "lon": lon, "urban": 0.75, "population": 0.7, "critical": 0.5},
            {"province": "Chiang Rai", "district": "Chiang Saen", "lat": 20.27, "lon": 100.08, "urban": 0.45, "population": 0.4, "critical": 0.2},
        ]
    return [
        {"province": "Thailand", "district": event.display_name.split(",")[0], "lat": lat, "lon": lon, "urban": 0.6, "population": 0.6, "critical": 0.3}
    ]
