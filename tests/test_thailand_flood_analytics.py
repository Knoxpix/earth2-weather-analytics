from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from thailand_flood_analytics.event_config import get_event, load_events
from thailand_flood_analytics.flood_risk_model import RiskInputs, compute_risk_score
from thailand_flood_analytics.validation import binary_metrics
from thailand_flood_analytics.warning_levels import WarningLevel, classify_warning_level, info_for


def test_event_config_loading():
    events = load_events()
    assert {"maesai_flood_2024", "hatyai_flood_2025", "dianmu_flood_2021", "noru_flood_2022"} <= set(events)
    assert events["hatyai_flood_2025"].aoi["lat_min"] == 5.5


def test_date_range_filtering():
    event = get_event("maesai_flood_2024")
    start, end = event.filter_range("2024-09-01", "2024-09-30")
    assert start == event.start_time
    assert end == event.end_time


def test_warning_thresholds_and_colors():
    assert classify_warning_level(0) == WarningLevel.GREEN
    assert classify_warning_level(20) == WarningLevel.YELLOW
    assert classify_warning_level(40) == WarningLevel.ORANGE
    assert classify_warning_level(60) == WarningLevel.RED
    assert classify_warning_level(80) == WarningLevel.PURPLE
    assert info_for(WarningLevel.PURPLE).color == "#8E44AD"


def test_direct_escalation_rules():
    score, level, drivers = compute_risk_score(
        RiskInputs(
            rainfall_24h=70,
            rainfall_72h=120,
            flood_depth=1.2,
            urban_exposure=0.9,
            population_exposure=0.8,
            critical_infrastructure_exposure=0.1,
        )
    )
    assert score >= 0
    assert level in {WarningLevel.RED, WarningLevel.PURPLE}
    assert "urban_flood_depth_ge_1m" in drivers


def test_validation_metrics():
    metrics = binary_metrics({"a", "b", "c"}, {"b", "c", "d"})
    assert round(metrics["iou"], 2) == 0.5
    assert round(metrics["precision"], 2) == 0.67
    assert round(metrics["recall"], 2) == 0.67


def test_cli_build_replay_outputs(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "thailand_flood_analytics.cli",
        "build-replay",
        "--event-id",
        "maesai_flood_2024",
        "--start",
        "2024-09-13",
        "--end",
        "2024-09-15",
        "--mode",
        "SYNTHETIC_DEMO_FALLBACK",
        "--output",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    summary = json.loads(result.stdout)
    manifest = Path(summary["manifest"])
    badges = Path(summary["warning_badges"])
    assert manifest.exists()
    assert badges.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["options"]["utc_start_time"].startswith("2024-09-13")
    assert data["thailand_flood_command_center"]["data_mode"] == "SYNTHETIC_DEMO_FALLBACK"
    assert all("*" not in json.dumps(feature) for feature in data["features"])
    image_sources = [
        source
        for feature in data["features"]
        if feature["type"] == "Image"
        for source in feature["sources"].values()
    ]
    assert image_sources
    assert all(source.endswith(".jpg") for source in image_sources)
    assert all((tmp_path / source).exists() for source in image_sources)


def test_metadata_contains_badge_legend(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "thailand_flood_analytics.cli",
        "build-replay",
        "--event-id",
        "hatyai_flood_2025",
        "--start",
        "2025-11-24",
        "--end",
        "2025-11-24",
        "--mode",
        "SYNTHETIC_DEMO_FALLBACK",
        "--output",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=True)
    manifest = tmp_path / "thailand_flood_command_center_hatyai_flood_2025.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    legend = data["thailand_flood_command_center"]["legend"]
    assert legend["GREEN"]["color"] == "#2ECC71"
    assert legend["PURPLE"]["label_en"] == "Critical / Evacuation Preparedness"
