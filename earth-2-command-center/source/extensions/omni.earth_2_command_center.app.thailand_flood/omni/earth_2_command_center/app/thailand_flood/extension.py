from __future__ import annotations

import asyncio
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import carb
import omni.ext
import omni.kit.async_engine as async_engine
import omni.ui as ui
from omni.kit.window.filepicker.datetime import DateWidget

from omni.earth_2_command_center.app.core import get_state


EVENTS = [
    ("maesai_flood_2024", "Mae Sai Flood 2024", "2024-09-13", "2024-09-19"),
    ("hatyai_flood_2025", "Hat Yai Flood 2025", "2025-11-17", "2025-11-28"),
    ("dianmu_flood_2021", "Dianmu Flood 2021", "2021-09-23", "2021-10-15"),
    ("noru_flood_2022", "Noru Flood 2022", "2022-09-28", "2022-10-15"),
]

LAYER_LABELS = [
    "Forecast rainfall",
    "Accumulated rainfall",
    "Flood risk heatmap",
    "Warning badges",
    "Satellite flood extent / observed water extent",
    "Province/district boundaries",
    "Rivers / watersheds",
    "Critical infrastructure",
]


class ThailandFloodPanel(ui.Window):
    def __init__(self, ext: "ThailandFloodExtension") -> None:
        super().__init__(
            "Thailand Flood Analytics Panel",
            width=390,
            height=0,
            flags=ui.WINDOW_FLAGS_NO_CLOSE,
        )
        self._ext = ext
        self._event_index = 0
        self._status = ui.SimpleStringModel("Ready")
        self._layer_models = [ui.SimpleBoolModel(True) for _ in LAYER_LABELS]
        self.frame.set_build_fn(self._build)

    def _build(self) -> None:
        with ui.VStack(spacing=6):
            ui.Label("Thailand Extreme Weather & Flood Analytics Command Center", word_wrap=True)
            ui.Separator()
            with ui.HStack(height=24):
                ui.Label("Event", width=110)
                self._event_combo = ui.ComboBox(0, *[item[1] for item in EVENTS])
                self._event_combo.model.add_item_changed_fn(self._on_event_changed)
            event = EVENTS[self._event_index]
            with ui.HStack(height=24):
                ui.Label("Start date", width=110)
                self._start_date = DateWidget()
                self._start_date.model.set_value(event[2])
            with ui.HStack(height=24):
                ui.Label("End date", width=110)
                self._end_date = DateWidget()
                self._end_date.model.set_value(event[3])
            with ui.HStack(height=24):
                ui.Label("Forecast init", width=110)
                self._init_combo = ui.ComboBox(0, "Default event init", "Earliest init", "Latest init")
            ui.Separator()
            ui.Label("Layers")
            for idx, label in enumerate(LAYER_LABELS):
                with ui.HStack(height=22):
                    ui.CheckBox(self._layer_models[idx], width=20)
                    ui.Label(label)
            ui.Separator()
            self._build_legend()
            ui.Separator()
            with ui.HStack(height=26):
                ui.Button("Run pipeline", clicked_fn=self._run_pipeline)
                ui.Button("Load cached replay", clicked_fn=self._load_cached)
            with ui.HStack(height=26):
                ui.Button("Focus timeline", clicked_fn=self._focus_timeline)
                ui.Button("Validation summary", clicked_fn=self._show_validation)
            ui.Label(self._status, word_wrap=True)

    def _build_legend(self) -> None:
        colors = [
            ("GREEN", 0xFF71CC2E),
            ("YELLOW", 0xFF0FC4F1),
            ("ORANGE", 0xFF227EE6),
            ("RED", 0xFF3C4CE7),
            ("PURPLE", 0xFFAD448E),
        ]
        with ui.HStack(height=22):
            ui.Label("Warning legend", width=110)
            for label, color in colors:
                with ui.HStack(width=50):
                    ui.Rectangle(width=14, height=14, style={"background_color": color})
                    ui.Label(label[:1], width=20)

    def _on_event_changed(self, model, item) -> None:
        self._event_index = model.get_item_value_model().as_int
        event = EVENTS[self._event_index]
        self._start_date.model.set_value(event[2])
        self._end_date.model.set_value(event[3])

    def _selected_event(self) -> tuple[str, str, str, str]:
        return EVENTS[self._event_index]

    def _date_value(self, widget: DateWidget) -> str:
        return f"{widget.model.year:04d}-{widget.model.month:02d}-{widget.model.day:02d}"

    def _run_pipeline(self) -> None:
        async_engine.run_coroutine(self._ext.build_and_load_replay(self._selected_event()[0], self._date_value(self._start_date), self._date_value(self._end_date)))

    def _load_cached(self) -> None:
        async_engine.run_coroutine(self._ext.load_cached_replay(self._selected_event()[0]))

    def _focus_timeline(self) -> None:
        self._ext.focus_timeline(self._date_value(self._start_date), self._date_value(self._end_date))
        self._status.set_value("Timeline focused on selected replay window.")

    def _show_validation(self) -> None:
        self._status.set_value("Validation summary is written by the CLI when prediction and ground truth files are provided.")

    def set_status(self, text: str) -> None:
        self._status.set_value(text)


class ThailandFloodExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._repo_root = self._find_repo_root()
        self._output_dir = Path(os.environ.get("E2CC_OUTPUT_DIR", str(self._repo_root / "outputs" / "e2cc")))
        self._window = ThailandFloodPanel(self)

    def on_shutdown(self) -> None:
        if self._window:
            self._window.destroy()
            self._window = None

    async def build_and_load_replay(self, event_id: str, start: str, end: str) -> None:
        self._window.set_status(f"Building replay for {event_id}...")
        cmd = [
            sys.executable,
            "-m",
            "thailand_flood_analytics.cli",
            "build-replay",
            "--event-id",
            event_id,
            "--mode",
            "auto",
            "--output",
            str(self._output_dir),
        ]
        if start:
            cmd.extend(["--start", start])
        if end:
            cmd.extend(["--end", end])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = await asyncio.to_thread(subprocess.run, cmd, cwd=str(self._repo_root), env=env, text=True, capture_output=True, check=True)
            summary = json.loads(proc.stdout)
            self.load_manifest(Path(summary["manifest"]))
            self._window.set_status(f"Loaded {event_id}: {summary['data_mode']}")
        except Exception as exc:
            carb.log_error(f"Thailand flood replay failed: {exc}")
            self._window.set_status(f"Replay failed: {exc}")

    async def load_cached_replay(self, event_id: str) -> None:
        manifest = self._output_dir / f"thailand_flood_command_center_{event_id}.json"
        if not manifest.exists():
            await self.build_and_load_replay(event_id, "", "")
            return
        self.load_manifest(manifest)
        self._window.set_status(f"Loaded cached replay: {manifest}")

    def load_manifest(self, manifest: Path) -> None:
        try:
            from omni.earth_2_command_center.app.test_sequence import get_ext
            from omni.earth_2_command_center.app.test_sequence.metadata_sequences import add_from_meta_json

            add_from_meta_json(get_ext(), str(manifest))
        except Exception as exc:
            carb.log_error(f"Could not load E2CC metadata manifest {manifest}: {exc}")
            raise

    def focus_timeline(self, start: str, end: str) -> None:
        start_dt = datetime.datetime.fromisoformat(start).replace(tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime.fromisoformat(end).replace(hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc)
        get_state().get_time_manager().set_time_range(start_dt, end_dt, playback_duration=30)
    def _find_repo_root(self) -> Path:
        for parent in Path(__file__).resolve().parents:
            if (parent / "configs" / "thailand_events.yaml").exists():
                return parent
        return Path.cwd()
