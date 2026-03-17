"""Frontend and engine listen settings."""

from __future__ import annotations

from pathlib import Path

from fastapiex.settings import BaseSettings, Settings

_UNSET = object()


class _BaseServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    uds: Path | None = None
    workers: int = 1
    backlog: int = 1024
    workers_kill_timeout: int | None = 30
    log_access: bool = False


@Settings("frontend.server")
class FrontendServerSettings(_BaseServerSettings):
    pass


@Settings("engine.server")
class EngineServerSettings(_BaseServerSettings):
    port: int = 9001



def apply_server_overrides(
    *,
    host: str | object = _UNSET,
    port: int | object = _UNSET,
    uds: Path | None | object = _UNSET,
    log_access: bool | object = _UNSET,
) -> None:
    from vfront.bootstrap import apply_settings_overrides

    overrides: dict[str, object | None] = {}
    if host is not _UNSET:
        overrides["engine.server.host"] = host
    if port is not _UNSET:
        overrides["engine.server.port"] = port
    if uds is not _UNSET:
        overrides["engine.server.uds"] = uds
    if log_access is not _UNSET:
        overrides["engine.server.log_access"] = log_access
    if overrides:
        apply_settings_overrides(overrides)
