"""Helpers for managed engine-worker processes and UDS endpoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from fastapiex.settings import BaseSettings, GetSettings, Settings

from vfront.shared.config.models import BackendConfig, RuntimeModelConfig

if TYPE_CHECKING:
    from vfront.shared.config.managed_workers import ManagedWorkerSettings

_SOCKET_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_UNSET = object()


@Settings("engine_worker")
class EngineWorkerSettings(BaseSettings):
    runtime_model: str | None = None


def apply_engine_worker_overrides(
    *,
    runtime_model: str | None | object = _UNSET,
) -> None:
    from vfront.bootstrap import apply_settings_overrides

    overrides: dict[str, object | None] = {}
    if runtime_model is not _UNSET:
        overrides["engine_worker.runtime_model"] = runtime_model
    if overrides:
        apply_settings_overrides(overrides)


def is_engine_worker_process() -> bool:
    """Whether the current process is serving a single managed engine worker."""
    return GetSettings("engine_worker").runtime_model is not None


def select_runtime_models(
    runtime_models: list[RuntimeModelConfig],
    worker_runtime_model: str | None,
) -> list[RuntimeModelConfig]:
    """Return the runtime models that this process should boot locally."""
    if worker_runtime_model is None:
        return runtime_models

    selected = [model for model in runtime_models if model.id == worker_runtime_model]
    if not selected:
        raise RuntimeError(
            f"No runtime model is configured for engine worker '{worker_runtime_model}'."
        )
    return selected


def managed_worker_socket_path(settings: ManagedWorkerSettings, runtime_model: str) -> Path:
    """Stable UDS path for a managed engine worker."""
    safe_name = _SOCKET_SAFE_RE.sub("-", runtime_model).strip("-.") or "engine"
    return settings.socket_dir / f"{safe_name}.sock"


def managed_worker_backend(
    worker_settings: ManagedWorkerSettings,
    *,
    backend_id: str,
    runtime_model: str,
) -> BackendConfig:
    """Synthetic backend definition for a managed local engine worker."""
    return BackendConfig(
        id=backend_id,
        base_url=f"unix://{managed_worker_socket_path(worker_settings, runtime_model)}",
        api_key=worker_settings.api_key,
        timeout_seconds=worker_settings.timeout_seconds,
    )
