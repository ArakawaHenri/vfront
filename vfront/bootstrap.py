"""Common bootstrap helpers for CLI entrypoints and app factories."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any

from fastapiex.settings import init_settings

_SETTINGS_DOMAIN_ROOTS = (
    "vfront.shared",
    "vfront.engine",
    "vfront.frontend",
)


def project_settings_path() -> Path:
    return Path(__file__).resolve().parents[1] / "settings.yaml"


def resolve_settings_path(settings_path: str | Path | None = None) -> Path:
    if settings_path is None:
        return project_settings_path()
    return Path(settings_path).resolve()


def _register_settings_domains() -> None:
    """Import domain roots so settings register via their package LCAs."""
    for module_name in _SETTINGS_DOMAIN_ROOTS:
        import_module(module_name)


def initialize_settings(settings_path: str | Path | None = None) -> Path:
    resolved = resolve_settings_path(settings_path)
    _register_settings_domains()
    init_settings(settings_path=resolved)
    return resolved


def apply_settings_overrides(overrides: dict[str, Any]) -> None:
    for path, value in overrides.items():
        env_key = "__".join(part.upper() for part in path.split("."))
        if value is None:
            os.environ.pop(env_key, None)
        else:
            if isinstance(value, os.PathLike):
                value = Path(value).resolve()
            os.environ[env_key] = str(value)
