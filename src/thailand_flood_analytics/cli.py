from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .data_sources import resolve_data_mode
from .e2cc_metadata_writer import write_outputs
from .event_config import get_event, load_events
from .geospatial_processing import synthetic_event_grid
from .validation import validate_files


def default_output_dir() -> Path:
    return Path(os.environ.get("E2CC_OUTPUT_DIR", "outputs/e2cc")).expanduser()


def cmd_list_events(args: argparse.Namespace) -> int:
    for event in load_events(args.config).values():
        print(f"{event.event_id}\t{event.display_name}\t{event.date_range['start']} -> {event.date_range['end']}")
    return 0


def build_replay(args: argparse.Namespace) -> dict[str, str]:
    event = get_event(args.event_id, args.config)
    start, end = event.filter_range(args.start, args.end)
    mode, details = resolve_data_mode(event, args.mode)
    grid = synthetic_event_grid(event, start, end)
    paths = write_outputs(event, grid, mode, Path(args.output))
    run_summary = {
        "event_id": event.event_id,
        "display_name": event.display_name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "data_mode": mode.value,
        "mode_details": details,
        "manifest": str(paths["manifest"]),
        "warning_badges": str(paths["badges"]),
        "safety_note": "This is a research/demo decision-support visualization, not an official warning system.",
    }
    summary_path = Path(args.output) / event.event_id / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return {k: str(v) for k, v in paths.items()}


def cmd_build_replay(args: argparse.Namespace) -> int:
    build_replay(args)
    return 0


def cmd_export_e2cc(args: argparse.Namespace) -> int:
    requested_output = Path(args.output)
    args.start = args.start or None
    args.end = args.end or None
    args.mode = args.mode or "auto"
    args.output = str(requested_output.parent if requested_output.suffix == ".json" else requested_output)
    paths = build_replay(args)
    if requested_output.suffix == ".json":
        requested_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths["manifest"], requested_output)
        print(str(requested_output))
    else:
        print(paths["manifest"])
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    paths = validate_files(args.event_id, Path(args.prediction), Path(args.ground_truth), Path(args.output))
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thailand-flood-analytics")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    list_events = sub.add_parser("list-events")
    list_events.set_defaults(func=cmd_list_events)

    build = sub.add_parser("build-replay")
    build.add_argument("--event-id", required=True)
    build.add_argument("--start", default=None)
    build.add_argument("--end", default=None)
    build.add_argument("--mode", default="auto")
    build.add_argument("--output", default=str(default_output_dir()))
    build.set_defaults(func=cmd_build_replay)

    validate = sub.add_parser("validate")
    validate.add_argument("--event-id", required=True)
    validate.add_argument("--prediction", required=True)
    validate.add_argument("--ground-truth", required=True)
    validate.add_argument("--output", default="outputs")
    validate.set_defaults(func=cmd_validate)

    export = sub.add_parser("export-e2cc")
    export.add_argument("--event-id", required=True)
    export.add_argument("--start", default=None)
    export.add_argument("--end", default=None)
    export.add_argument("--mode", default="auto")
    export.add_argument("--output", default=str(default_output_dir()))
    export.set_defaults(func=cmd_export_e2cc)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
