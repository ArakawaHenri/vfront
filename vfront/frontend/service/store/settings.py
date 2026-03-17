"""Persistent KV store settings."""

from __future__ import annotations

from pathlib import Path

from fastapiex.settings import BaseSettings, Settings
from pydantic import Field


@Settings("frontend.store")
class StoreSettings(BaseSettings):
    path: Path = Field(default_factory=lambda: Path("./data/store.lmdb"))
    map_size_mb: int = 512
    max_dbs: int = 64
    max_readers: int = 256
    sync: bool = True
    metasync: bool = True
    writemap: bool = True
    map_async: bool = False
    max_key_bytes: int = 256
    max_value_bytes: int = 100 * 1024 * 1024
    cleanup_interval_seconds: int = 60
    cleanup_max_deletes: int = 1_000_000
    default_retention_minutes: int = 1440
