"""Engine-side settings for the internal worker API."""

from __future__ import annotations

from fastapiex.settings import BaseSettings, Settings

_UNSET = object()


@Settings("engine.api")
class EngineAPISettings(BaseSettings):
    api_key: str | None = None
    log_access: bool = False



def apply_engine_api_overrides(
    *,
    api_key: str | None | object = _UNSET,
    log_access: bool | object = _UNSET,
) -> None:
    from vfront.bootstrap import apply_settings_overrides

    overrides: dict[str, object | None] = {}
    if api_key is not _UNSET:
        overrides["engine.api.api_key"] = api_key
    if log_access is not _UNSET:
        overrides["engine.api.log_access"] = log_access
    if overrides:
        apply_settings_overrides(overrides)
