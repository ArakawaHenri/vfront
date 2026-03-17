"""Centralized validation for deployment mode settings."""

from __future__ import annotations

from typing import Protocol


class _HasModels(Protocol):
    @property
    def models(self) -> list[object]: ...


class _HasTransport(Protocol):
    @property
    def transport(self) -> str: ...


class _HasWorkers(Protocol):
    @property
    def workers(self) -> int: ...


class _HasModelsAndTransport(_HasModels, _HasTransport, Protocol):
    pass


def validate_frontend_mode_settings(routing_settings: _HasModelsAndTransport) -> None:
    if not routing_settings.models:
        raise RuntimeError(
            "frontend mode requires frontend.routing.models to contain at least one public model."
        )
    if routing_settings.transport != "remote":
        raise RuntimeError(
            "frontend mode is control-plane only; "
            "use 'vfront embedded' or set frontend.routing.transport to 'remote'."
        )


def validate_engine_mode_settings(runtime_settings: _HasModels) -> None:
    if not runtime_settings.models:
        raise RuntimeError(
            "engine mode requires engine.runtime.models to contain at least one runtime model."
        )


def validate_embedded_mode_settings(
    *,
    routing_settings: _HasModelsAndTransport,
    runtime_settings: _HasModels,
    server_settings: _HasWorkers,
) -> bool:
    if not routing_settings.models:
        raise RuntimeError(
            "embedded mode requires frontend.routing.models to contain at least one public model."
        )
    if not runtime_settings.models:
        raise RuntimeError(
            "embedded mode requires engine.runtime.models to contain at least one runtime model."
        )

    use_private_runner = routing_settings.transport != "local"
    if not use_private_runner and server_settings.workers != 1:
        raise RuntimeError(
            "frontend.routing.transport=local requires frontend.server.workers=1 because the "
            "embedded job runner and in-process engine are process-local."
        )
    return use_private_runner
