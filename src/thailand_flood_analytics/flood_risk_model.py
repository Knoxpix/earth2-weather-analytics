from __future__ import annotations

from dataclasses import dataclass

from .warning_levels import WarningLevel, apply_escalation_rules, classify_warning_level


@dataclass(frozen=True)
class RiskInputs:
    rainfall_24h: float
    rainfall_72h: float
    river_level_percentile: float = 0.0
    flood_depth: float = 0.0
    flood_velocity: float = 0.0
    terrain_susceptibility: float = 0.5
    population_exposure: float = 0.5
    urban_exposure: float = 0.5
    critical_infrastructure_exposure: float = 0.0
    historical_recurrence: float = 0.5
    observed_major_inundation: bool = False


def clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    return max(min_value, min(max_value, float(value)))


def compute_risk_score(inputs: RiskInputs) -> tuple[float, WarningLevel, list[str]]:
    rainfall_24 = clamp(inputs.rainfall_24h / 250.0 * 100.0)
    rainfall_72 = clamp(inputs.rainfall_72h / 500.0 * 100.0)
    river = clamp(inputs.river_level_percentile * 100.0)
    depth = clamp(inputs.flood_depth / 2.0 * 100.0)
    velocity = clamp(inputs.flood_velocity / 2.0 * 100.0)
    terrain = clamp(inputs.terrain_susceptibility * 100.0)
    population = clamp(inputs.population_exposure * 100.0)
    urban = clamp(inputs.urban_exposure * 100.0)
    critical = clamp(inputs.critical_infrastructure_exposure * 100.0)
    recurrence = clamp(inputs.historical_recurrence * 100.0)

    score = (
        rainfall_24 * 0.18
        + rainfall_72 * 0.18
        + river * 0.15
        + depth * 0.16
        + velocity * 0.05
        + terrain * 0.06
        + population * 0.08
        + urban * 0.07
        + critical * 0.04
        + recurrence * 0.03
    )
    level = classify_warning_level(score)
    drivers: list[str] = []
    if inputs.rainfall_24h >= 100:
        drivers.append("rainfall_24h_high")
    if inputs.rainfall_72h >= 250:
        drivers.append("rainfall_72h_high")
    if inputs.urban_exposure >= 0.7:
        drivers.append("urban_exposure_high")
    if inputs.critical_infrastructure_exposure >= 0.5:
        drivers.append("critical_infrastructure_exposed")

    level, escalation_drivers = apply_escalation_rules(
        level,
        flood_depth_m=inputs.flood_depth,
        urban_area=inputs.urban_exposure >= 0.6,
        critical_infrastructure_exposed=inputs.critical_infrastructure_exposure >= 0.5,
        population_exposure_high=inputs.population_exposure >= 0.7,
        rainfall_24h_mm=inputs.rainfall_24h,
        river_level_percentile=inputs.river_level_percentile,
        observed_major_inundation=inputs.observed_major_inundation,
    )
    return round(clamp(score), 2), level, drivers + escalation_drivers
