"""Application metadata settings."""

from __future__ import annotations

from pathlib import Path

from fastapiex.settings import BaseSettings, Settings
from pydantic import Field


class _BaseAppSettings(BaseSettings):
    title: str = "vfront"
    version: str = "0.1.0"
    debug_mode: bool = False
    log_dir: Path = Field(default_factory=lambda: Path("./logs"))
    cors_origins: list[str] = Field(default_factory=list)
    api_key: str | None = None


@Settings("frontend.app")
class FrontendAppSettings(_BaseAppSettings):
    description: str = "OpenAI-compatible frontend for vLLM"


@Settings("engine.app")
class EngineAppSettings(_BaseAppSettings):
    description: str = "Dedicated vLLM engine worker"
