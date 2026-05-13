from __future__ import annotations

from .data_sources import DataMode
from .event_config import EventConfig


def describe_runner(event: EventConfig, mode: DataMode) -> dict[str, str]:
    return {
        "event_id": event.event_id,
        "requested_execution_mode": mode.value,
        "note": "Real Earth2Studio execution is integrated through cached/local outputs in this demo CLI; direct inference can be added behind this adapter.",
    }
