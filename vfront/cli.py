"""Public CLI for frontend, engine, and embedded deployment modes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vfront.bootstrap import resolve_settings_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vfront")
    subparsers = parser.add_subparsers(dest="command", required=True)

    frontend_parser = subparsers.add_parser("frontend")
    embedded_parser = subparsers.add_parser("embedded")
    engine_parser = subparsers.add_parser("engine")

    for subparser in (frontend_parser, embedded_parser, engine_parser):
        subparser.add_argument(
            "-c",
            "--config",
            dest="config",
            default=None,
            help="Path to the YAML settings file.",
        )

    engine_parser.add_argument(
        "--runtime-model",
        dest="runtime_model",
        default=None,
        help="Runtime model id to serve from engine.runtime.models.",
    )

    return parser
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings_path = resolve_settings_path(args.config)

    if args.command == "frontend":
        from vfront.modes.frontend import run_frontend

        run_frontend(settings_path)
    elif args.command == "embedded":
        from vfront.modes.embedded import run_embedded

        run_embedded(settings_path)
    else:
        from vfront.modes.engine import run_engine

        run_engine(settings_path, args.runtime_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
