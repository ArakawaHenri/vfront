"""Runtime helpers for embedded versus managed job-runner modes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapiex.settings import GetSettings

if TYPE_CHECKING:
    from vfront.frontend.service.jobs.settings import JobSettings

_IS_INTERNAL_RUNNER_PROCESS = False


def configure_internal_runner_process(enabled: bool) -> None:
    global _IS_INTERNAL_RUNNER_PROCESS
    _IS_INTERNAL_RUNNER_PROCESS = enabled


def is_internal_runner_process() -> bool:
    """Whether this process is the private runner subprocess."""
    return _IS_INTERNAL_RUNNER_PROCESS


def should_run_job_runner(settings: JobSettings) -> bool:
    """Return whether this process should own the job runner service."""
    if not settings.enabled:
        return False
    if is_internal_runner_process():
        return True
    try:
        routing_settings = GetSettings("frontend.routing")
    except Exception:
        return settings.run_embedded
    return settings.run_embedded and routing_settings.transport == "local"
