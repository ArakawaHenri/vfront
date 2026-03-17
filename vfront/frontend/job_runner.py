"""Internal entrypoint for the private frontend background job runner."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI
from fastapiex.di import install_di
from fastapiex.settings import GetSettings

from vfront.bootstrap import initialize_settings
from vfront.core.logging import setup_logging, shutdown_logging
from vfront.frontend.service.jobs.runner import JobRunnerService
from vfront.frontend.service.jobs.runtime import configure_internal_runner_process


async def _run(settings_path: str | None = None) -> None:
    initialize_settings(settings_path)
    app_settings = GetSettings("frontend.app")
    setup_logging(log_dir=app_settings.log_dir, debug=app_settings.debug_mode)

    app = FastAPI()
    install_di(app, service_packages=["vfront.frontend.service", "vfront.frontend.remote_service"])
    try:
        async with app.router.lifespan_context(app):
            _ = JobRunnerService
            await asyncio.Event().wait()
    finally:
        await shutdown_logging()


def run_job_runner(settings_path: str | None = None) -> None:
    configure_internal_runner_process(True)
    asyncio.run(_run(settings_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vfront-frontend-job-runner")
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        help="Path to the YAML settings file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings_path = None if args.config is None else str(Path(args.config).resolve())
    run_job_runner(settings_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
