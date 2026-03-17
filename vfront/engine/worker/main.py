"""Entrypoint for a single engine-worker process."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fastapiex.settings import GetSettings
from granian import Granian
from granian.constants import Interfaces, Loops

from vfront.bootstrap import initialize_settings
from vfront.shared.config.server import apply_server_overrides
from vfront.shared.engine.managed import apply_engine_worker_overrides


def run_engine_worker(settings_path: str | None = None) -> None:
    initialize_settings(settings_path)

    runtime_model = GetSettings("engine_worker").runtime_model
    server_settings = GetSettings("engine.server")
    engine_api_settings = GetSettings("engine.api")
    effective_socket_path = server_settings.uds
    if effective_socket_path is not None:
        effective_socket_path.parent.mkdir(parents=True, exist_ok=True)
        if effective_socket_path.exists():
            effective_socket_path.unlink()
        Granian(
            "vfront.engine.worker.app:app",
            address=server_settings.host,
            port=0,
            uds=effective_socket_path,
            interface=Interfaces.ASGI,
            workers=1,
            backlog=server_settings.backlog,
            workers_kill_timeout=server_settings.workers_kill_timeout,
            log_access=engine_api_settings.log_access,
            loop=Loops.uvloop,
            process_name=(
                f"vfront-engine:{runtime_model}" if runtime_model else "vfront-engine"
            ),
        ).serve()
    else:
        Granian(
            "vfront.engine.worker.app:app",
            address=server_settings.host,
            port=server_settings.port,
            interface=Interfaces.ASGI,
            workers=1,
            backlog=server_settings.backlog,
            workers_kill_timeout=server_settings.workers_kill_timeout,
            log_access=engine_api_settings.log_access,
            loop=Loops.uvloop,
            process_name="vfront-engine",
        ).serve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vfront-engine-worker")
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        help="Path to the YAML settings file.",
    )
    parser.add_argument(
        "--runtime-model",
        dest="runtime_model",
        default=None,
        help="Runtime model id to serve from engine.runtime.models.",
    )
    parser.add_argument(
        "--uds",
        dest="uds",
        default=None,
        help="Optional UDS socket path for this worker.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    apply_engine_worker_overrides(runtime_model=args.runtime_model)
    apply_server_overrides(uds=None if args.uds is None else Path(args.uds))
    settings_path = None if args.config is None else str(Path(args.config).resolve())
    run_engine_worker(settings_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
