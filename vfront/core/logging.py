"""Loguru-based logging with request-id context propagation.

Pattern borrowed from neo_tx_classifier:
- ContextVar for request-id, injected into all log records
- InterceptHandler bridges stdlib logging → Loguru
- Dual sinks: console (colored) + file (daily rotation)
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import sys
from pathlib import Path
from typing import Final

from loguru import logger as _loguru_logger

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="-",
)

_LOGGING_CONFIGURED: bool = False
_CONSOLE_SINK_ID: int | None = None
_FILE_SINK_ID: int | None = None
_THIS_FILE: Final[str] = __file__


def _patch_record(record: dict) -> None:
    """Inject common context into all loguru records."""
    record["extra"]["request_id"] = request_id_ctx.get("-")


_loguru_logger.configure(patcher=_patch_record)  # type: ignore[arg-type]
logger = _loguru_logger


class InterceptHandler(logging.Handler):
    """Forward standard logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 1
        while frame is not None:
            filename = frame.f_code.co_filename
            module_name = frame.f_globals.get("__name__", "")
            if filename == _THIS_FILE or module_name.startswith("logging"):
                frame = frame.f_back  # type: ignore[assignment]
                depth += 1
                continue
            break

        extra = {"logger_name": record.name}
        builtin_keys = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
        }
        for key, value in record.__dict__.items():
            if key not in builtin_keys and key not in extra:
                extra[key] = value

        log = logger.bind(**extra)
        message = record.getMessage()
        if record.stack_info:
            message = f"{message}\n{record.stack_info}"
        log.opt(depth=depth, exception=record.exc_info).log(level, message)


def _configure_std_logging(level: int) -> None:
    """Take over standard logging."""
    intercept_handler = InterceptHandler()
    root = logging.getLogger()
    root.handlers = [intercept_handler]
    root.setLevel(level)
    logging.captureWarnings(True)


def setup_logging(log_dir: Path, debug: bool) -> None:
    """Initialize process-wide logging. Repeated calls overwrite previous config."""
    global _LOGGING_CONFIGURED, _CONSOLE_SINK_ID, _FILE_SINK_ID

    log_dir.mkdir(parents=True, exist_ok=True)
    level_name = "DEBUG" if debug else "INFO"
    level_no = logging.DEBUG if debug else logging.INFO

    if _CONSOLE_SINK_ID is not None:
        with contextlib.suppress(Exception):
            logger.remove(_CONSOLE_SINK_ID)
        _CONSOLE_SINK_ID = None

    if _FILE_SINK_ID is not None:
        with contextlib.suppress(Exception):
            logger.remove(_FILE_SINK_ID)
        _FILE_SINK_ID = None

    if not _LOGGING_CONFIGURED:
        _loguru_logger.remove()

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<magenta>{extra[request_id]}</magenta> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{extra[request_id]} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    _CONSOLE_SINK_ID = logger.add(
        sys.stderr,
        level=level_name,
        colorize=True,
        format=console_format,
        backtrace=debug,
        diagnose=debug,
        enqueue=False,
    )

    _FILE_SINK_ID = logger.add(
        log_dir / "vfront.log",
        level="DEBUG",
        format=file_format,
        rotation="00:00",
        retention="7 days",
        compression="zip",
        enqueue=True,
        backtrace=debug,
        diagnose=debug,
        encoding="utf-8",
    )

    _configure_std_logging(level_no)
    _LOGGING_CONFIGURED = True
    logger.bind(logger_name=__name__).debug("Logging configured.")


async def shutdown_logging(remove_sinks: bool = False) -> None:
    """Flush async logging queue before app shutdown."""
    await logger.complete()
    if remove_sinks:
        global _CONSOLE_SINK_ID, _FILE_SINK_ID
        if _CONSOLE_SINK_ID is not None:
            with contextlib.suppress(Exception):
                logger.remove(_CONSOLE_SINK_ID)
            _CONSOLE_SINK_ID = None
        if _FILE_SINK_ID is not None:
            with contextlib.suppress(Exception):
                logger.remove(_FILE_SINK_ID)
            _FILE_SINK_ID = None
