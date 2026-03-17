"""Filesystem-backed uploaded file settings."""

from __future__ import annotations

from pathlib import Path

from fastapiex.settings import BaseSettings, Settings
from pydantic import Field


@Settings("frontend.file_storage")
class FileStorageSettings(BaseSettings):
    base_dir: Path = Field(default_factory=lambda: Path("./data/files"))
    max_file_size_mb: int = 200
    retention_days: int = 30
    cleanup_interval_seconds: int = 300
