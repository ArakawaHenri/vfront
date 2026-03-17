"""Background job-runner settings."""

from __future__ import annotations

from fastapiex.settings import BaseSettings, Settings


@Settings("frontend.jobs")
class JobSettings(BaseSettings):
    enabled: bool = True
    run_embedded: bool = True
    poll_interval_seconds: float = 0.1
    lease_seconds: int = 300
    max_parallel_jobs: int = 8
    runner_id: str | None = None
