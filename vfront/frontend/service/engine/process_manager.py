"""Lifecycle manager for locally managed engine-worker subprocesses."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from vfront.shared.engine.http_client import resolve_http_endpoint
from vfront.shared.engine.managed import (
    managed_worker_backend,
    managed_worker_socket_path,
)

if TYPE_CHECKING:
    from vfront.frontend.service.engine.settings import RoutingSettings
    from vfront.shared.config.managed_workers import ManagedWorkerSettings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ManagedEngineProcess:
    """A managed engine worker plus ownership metadata."""

    runtime_model: str
    socket_path: Path
    process: subprocess.Popen[str] | None
    owned: bool
    log_path: Path | None = None


class ManagedEngineProcessManager:
    """Start and stop local engine workers when transport=managed."""

    def __init__(
        self,
        *,
        settings_path: Path,
        routing_settings: RoutingSettings,
        worker_settings: ManagedWorkerSettings,
    ) -> None:
        self._settings_path = settings_path
        self._routing_settings = routing_settings
        self._worker_settings = worker_settings
        self._processes: dict[str, ManagedEngineProcess] = {}

    def start(self) -> None:
        if self._routing_settings.transport != "managed":
            return

        self._worker_settings.socket_dir.mkdir(parents=True, exist_ok=True)
        for route in self._routing_settings.runtime_routes:
            self._ensure_worker(route.runtime_model)

    def shutdown(self) -> None:
        managed_processes = list(self._processes.values())
        self._processes.clear()
        for managed in managed_processes:
            if not managed.owned or managed.process is None:
                continue
            try:
                self._stop_process(managed.process)
                if managed.socket_path.exists():
                    managed.socket_path.unlink()
            except Exception:
                logger.exception(
                    "Failed to shut down managed engine worker '%s'.",
                    managed.runtime_model,
                )

    def _ensure_worker(self, runtime_model: str) -> None:
        backend = managed_worker_backend(
            self._worker_settings,
            backend_id=next(
                route.backend
                for route in self._routing_settings.runtime_routes
                if route.runtime_model == runtime_model
            ),
            runtime_model=runtime_model,
        )
        socket_path = managed_worker_socket_path(self._worker_settings, runtime_model)

        if self._is_ready(backend.base_url, backend.api_key):
            self._processes[runtime_model] = ManagedEngineProcess(
                runtime_model=runtime_model,
                socket_path=socket_path,
                process=None,
                owned=False,
            )
            return

        if socket_path.exists():
            socket_path.unlink()

        process = self._spawn(runtime_model)
        self._wait_until_ready(
            runtime_model=runtime_model,
            endpoint=backend.base_url,
            api_key=backend.api_key,
            process=process,
        )
        self._processes[runtime_model] = ManagedEngineProcess(
            runtime_model=runtime_model,
            socket_path=socket_path,
            process=process,
            owned=True,
            log_path=self._worker_log_path(runtime_model),
        )

    def _spawn(self, runtime_model: str) -> subprocess.Popen[str]:
        log_path = self._worker_log_path(runtime_model)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting managed engine worker '%s' with socket '%s' and startup log '%s'.",
            runtime_model,
            managed_worker_socket_path(self._worker_settings, runtime_model),
            log_path,
        )
        with log_path.open("w", encoding="utf-8") as startup_log:
            return subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vfront.engine.worker.main",
                    "--config",
                    str(self._settings_path),
                    "--runtime-model",
                    runtime_model,
                    "--uds",
                    str(managed_worker_socket_path(self._worker_settings, runtime_model)),
                ],
                cwd=str(self._settings_path.parent),
                text=True,
                stdout=startup_log,
                stderr=subprocess.STDOUT,
            )

    def _wait_until_ready(
        self,
        *,
        runtime_model: str,
        endpoint: str,
        api_key: str | None,
        process: subprocess.Popen[str],
    ) -> None:
        deadline = time.monotonic() + self._worker_settings.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                diagnostics = self._startup_diagnostics(runtime_model)
                logger.error(
                    "Managed engine worker '%s' exited before becoming ready. %s",
                    runtime_model,
                    diagnostics,
                )
                raise RuntimeError(
                    f"Managed engine worker '{runtime_model}' exited before becoming ready. "
                    f"{diagnostics}"
                )
            try:
                if self._is_ready(endpoint, api_key):
                    return
            except Exception as exc:  # pragma: no cover - defensive logging path
                last_error = exc
            time.sleep(0.2)

        stop_error: Exception | None = None
        try:
            self._stop_process(process)
        except Exception as exc:
            stop_error = exc
            logger.exception(
                "Failed to stop timed out managed engine worker '%s'.",
                runtime_model,
            )
        diagnostics = self._startup_diagnostics(runtime_model)
        logger.error(
            "Managed engine worker '%s' did not become ready within %s seconds. %s",
            runtime_model,
            self._worker_settings.startup_timeout_seconds,
            diagnostics,
        )
        raise RuntimeError(
            f"Managed engine worker '{runtime_model}' did not become ready within "
            f"{self._worker_settings.startup_timeout_seconds} seconds. {diagnostics}"
        ) from last_error or stop_error

    def _stop_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=self._worker_settings.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    @staticmethod
    def _is_ready(endpoint: str, api_key: str | None) -> bool:
        resolved = resolve_http_endpoint(endpoint)
        transport = None
        if resolved.uds is not None:
            transport = httpx.HTTPTransport(uds=resolved.uds)
        headers = {}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            with httpx.Client(
                base_url=resolved.base_url,
                transport=transport,
                headers=headers,
                timeout=1.0,
            ) as client:
                response = client.get("/health")
                response.raise_for_status()
                return bool(response.json().get("ready"))
        except Exception:
            return False

    def _worker_log_path(self, runtime_model: str) -> Path:
        return managed_worker_socket_path(
            self._worker_settings,
            runtime_model,
        ).with_suffix(".log")

    def _startup_diagnostics(self, runtime_model: str) -> str:
        log_path = self._worker_log_path(runtime_model)
        tail = self._read_log_tail(log_path)
        if tail is None:
            return f"No startup log was captured for '{runtime_model}'."
        if not tail:
            return f"Startup log is empty: {log_path}"
        return f"Startup log: {log_path}. Last log lines:\n{tail}"

    @staticmethod
    def _read_log_tail(log_path: Path, *, max_lines: int = 20) -> str | None:
        if not log_path.exists():
            return None
        try:
            with log_path.open("r", encoding="utf-8") as handle:
                lines = list(handle)
        except OSError:
            return f"<unable to read startup log at {log_path}>"
        tail_lines = lines[-max_lines:]
        return "".join(islice(tail_lines, 0, max_lines)).strip()
