"""Dedicated engine-worker deployment mode."""

from __future__ import annotations

from pathlib import Path

from fastapiex.settings import GetSettings

from vfront.bootstrap import initialize_settings
from vfront.engine.worker.main import run_engine_worker
from vfront.modes.validation import validate_engine_mode_settings
from vfront.shared.engine.managed import apply_engine_worker_overrides


def run_engine(settings_path: Path, runtime_model: str | None) -> None:
    apply_engine_worker_overrides(runtime_model=runtime_model)
    initialize_settings(settings_path)
    runtime_settings = GetSettings("engine.runtime")
    validate_engine_mode_settings(runtime_settings)
    run_engine_worker(str(settings_path))
