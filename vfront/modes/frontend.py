"""Frontend deployment mode."""

from __future__ import annotations

from pathlib import Path

from fastapiex.settings import GetSettings
from granian import Granian
from granian.constants import Interfaces, Loops

from vfront.bootstrap import initialize_settings
from vfront.frontend.service.jobs.process_manager import ManagedJobRunnerProcessManager
from vfront.modes.validation import validate_frontend_mode_settings


def run_frontend(settings_path: Path) -> None:
    initialize_settings(settings_path)

    server_settings = GetSettings("frontend.server")
    routing_settings = GetSettings("frontend.routing")
    job_settings = GetSettings("frontend.jobs")

    validate_frontend_mode_settings(routing_settings)

    runner_manager = ManagedJobRunnerProcessManager(
        settings_path=settings_path,
        job_settings=job_settings,
    )

    try:
        runner_manager.start()
        Granian(
            "vfront.frontend.main:app",
            address=server_settings.host,
            port=server_settings.port,
            interface=Interfaces.ASGI,
            workers=server_settings.workers,
            backlog=server_settings.backlog,
            workers_kill_timeout=server_settings.workers_kill_timeout,
            log_access=server_settings.log_access,
            loop=Loops.uvloop,
        ).serve()
    finally:
        runner_manager.shutdown()
