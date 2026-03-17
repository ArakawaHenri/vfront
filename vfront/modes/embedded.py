"""Embedded deployment mode."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from fastapiex.settings import GetSettings
from granian import Granian
from granian.constants import Interfaces, Loops

from vfront.bootstrap import initialize_settings
from vfront.frontend.service.engine.process_manager import ManagedEngineProcessManager
from vfront.frontend.service.jobs.process_manager import ManagedJobRunnerProcessManager
from vfront.modes.validation import validate_embedded_mode_settings


def _raise_mode_errors(message: str, errors: list[BaseException]) -> NoReturn:
    if len(errors) == 1:
        raise errors[0]
    exception_errors = [error for error in errors if isinstance(error, Exception)]
    if len(exception_errors) == len(errors):
        raise ExceptionGroup(message, exception_errors)
    raise BaseExceptionGroup(message, errors)


def run_embedded(settings_path: Path) -> None:
    initialize_settings(settings_path)

    server_settings = GetSettings("frontend.server")
    routing_settings = GetSettings("frontend.routing")
    runtime_settings = GetSettings("engine.runtime")
    worker_settings = GetSettings("engine.managed_workers")
    job_settings = GetSettings("frontend.jobs")

    use_private_runner = validate_embedded_mode_settings(
        routing_settings=routing_settings,
        runtime_settings=runtime_settings,
        server_settings=server_settings,
    )

    runner_manager = ManagedJobRunnerProcessManager(
        settings_path=settings_path,
        job_settings=job_settings,
    )
    engine_manager = ManagedEngineProcessManager(
        settings_path=settings_path,
        routing_settings=routing_settings,
        worker_settings=worker_settings,
    )

    pending_error: BaseException | None = None
    try:
        if use_private_runner:
            runner_manager.start()
        if routing_settings.transport == "managed":
            engine_manager.start()
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
    except BaseException as exc:
        pending_error = exc

    cleanup_errors: list[BaseException] = []
    try:
        engine_manager.shutdown()
    except BaseException as exc:
        cleanup_errors.append(exc)
    if use_private_runner:
        try:
            runner_manager.shutdown()
        except BaseException as exc:
            cleanup_errors.append(exc)

    if pending_error is not None:
        if cleanup_errors:
            _raise_mode_errors(
                "Embedded mode failed and shutdown cleanup also failed.",
                [pending_error, *cleanup_errors],
            )
        raise pending_error
    if cleanup_errors:
        _raise_mode_errors("Embedded mode shutdown failed.", cleanup_errors)
