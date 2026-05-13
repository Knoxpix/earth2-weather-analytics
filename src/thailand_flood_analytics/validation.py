from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def binary_metrics(predicted: set[Any], observed: set[Any]) -> dict[str, float]:
    tp = len(predicted & observed)
    fp = len(predicted - observed)
    fn = len(observed - predicted)
    union = len(predicted | observed)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "iou": tp / union if union else 0.0,
        "precision": precision,
        "recall": recall,
        "false_alarm_ratio": fp / (tp + fp) if tp + fp else 0.0,
        "missed_warning_ratio": fn / (tp + fn) if tp + fn else 0.0,
        "hit_rate": recall,
    }


def validate_files(event_id: str, prediction: Path, ground_truth: Path, output_dir: Path) -> dict[str, Path]:
    pred_data = json.loads(prediction.read_text(encoding="utf-8"))
    truth_data = json.loads(ground_truth.read_text(encoding="utf-8"))

    def ids(data: dict[str, Any]) -> set[str]:
        if "features" in data:
            return {str(f.get("id") or f.get("properties", {}).get("id") or idx) for idx, f in enumerate(data["features"])}
        return set(map(str, data.get("ids", [])))

    metrics = binary_metrics(ids(pred_data), ids(truth_data))
    out = output_dir / "validation"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / f"{event_id}_validation_summary.json"
    table = out / f"{event_id}_validation_table.csv"
    summary.write_text(json.dumps({"event_id": event_id, "metrics": metrics}, indent=2), encoding="utf-8")
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    return {"summary": summary, "table": table}
