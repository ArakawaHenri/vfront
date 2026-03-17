"""Shared settings for embedded managed engine workers."""

from __future__ import annotations

from pathlib import Path

from fastapiex.settings import BaseSettings, Settings
from pydantic import Field


@Settings("engine.managed_workers")
class ManagedWorkerSettings(BaseSettings):
    api_key: str | None = None
    socket_dir: Path = Field(default_factory=lambda: Path("/tmp/vfront"))
    timeout_seconds: float = 300.0
    startup_timeout_seconds: float = 60.0
    shutdown_timeout_seconds: int = 10
