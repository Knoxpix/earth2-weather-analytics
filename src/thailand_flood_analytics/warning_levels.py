from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class WarningInfo:
    code: str
    label_th: str
    label_en: str
    color: str
    score_min: float
    score_max: float
    meaning: str
    recommended_action: str


class WarningLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    PURPLE = "PURPLE"


WARNING_LEVELS: dict[WarningLevel, WarningInfo] = {
    WarningLevel.GREEN: WarningInfo(
        "GREEN",
        "ปกติ / ติดตามสถานการณ์",
        "Normal / Monitoring only",
        "#2ECC71",
        0,
        19,
        "Normal monitoring.",
        "Monitor weather and drainage conditions.",
    ),
    WarningLevel.YELLOW: WarningInfo(
        "YELLOW",
        "เฝ้าระวัง",
        "Watch",
        "#F1C40F",
        20,
        39,
        "Watch condition.",
        "Increase monitoring and prepare local response checks.",
    ),
    WarningLevel.ORANGE: WarningInfo(
        "ORANGE",
        "เสี่ยงปานกลาง",
        "Moderate risk",
        "#E67E22",
        40,
        59,
        "Moderate flood risk.",
        "Pre-position pumps, shelters, and response teams.",
    ),
    WarningLevel.RED: WarningInfo(
        "RED",
        "เสี่ยงสูง",
        "High risk",
        "#E74C3C",
        60,
        79,
        "High flood risk.",
        "Issue local preparedness alerts and evacuation advisories for low-lying areas.",
    ),
    WarningLevel.PURPLE: WarningInfo(
        "PURPLE",
        "วิกฤต / เตรียมอพยพ",
        "Critical / Evacuation Preparedness",
        "#8E44AD",
        80,
        100,
        "Critical flood risk.",
        "Prepare or execute evacuations where confirmed by authorized agencies.",
    ),
}


def info_for(level: WarningLevel | str) -> WarningInfo:
    return WARNING_LEVELS[WarningLevel(level)]


def classify_warning_level(risk_score: float) -> WarningLevel:
    score = max(0.0, min(100.0, float(risk_score)))
    if score < 20:
        return WarningLevel.GREEN
    if score < 40:
        return WarningLevel.YELLOW
    if score < 60:
        return WarningLevel.ORANGE
    if score < 80:
        return WarningLevel.RED
    return WarningLevel.PURPLE


def apply_escalation_rules(
    level: WarningLevel,
    *,
    flood_depth_m: float = 0.0,
    urban_area: bool = False,
    critical_infrastructure_exposed: bool = False,
    population_exposure_high: bool = False,
    rainfall_24h_mm: float = 0.0,
    river_level_percentile: float = 0.0,
    observed_major_inundation: bool = False,
) -> tuple[WarningLevel, list[str]]:
    severity_order = list(WarningLevel)
    drivers: list[str] = []

    def at_least(current: WarningLevel, minimum: WarningLevel, driver: str) -> WarningLevel:
        drivers.append(driver)
        return minimum if severity_order.index(current) < severity_order.index(minimum) else current

    if flood_depth_m >= 1.0 and urban_area:
        level = at_least(level, WarningLevel.RED, "urban_flood_depth_ge_1m")
    if critical_infrastructure_exposed:
        level = at_least(level, WarningLevel.RED, "critical_infrastructure_exposed")
    if flood_depth_m >= 1.5 and population_exposure_high:
        level = at_least(level, WarningLevel.PURPLE, "deep_flood_high_population_exposure")
    if rainfall_24h_mm >= 150 and river_level_percentile >= 0.9:
        level = at_least(level, WarningLevel.RED, "extreme_rainfall_river_near_overflow")
    elif rainfall_24h_mm >= 100 and river_level_percentile >= 0.8:
        level = at_least(level, WarningLevel.ORANGE, "heavy_rainfall_river_elevated")
    if observed_major_inundation:
        level = at_least(level, WarningLevel.RED, "observed_major_inundation")

    return level, drivers
