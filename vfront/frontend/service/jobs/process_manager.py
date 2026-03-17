"""Lifecycle manager for the private background job-runner subprocess."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vfront.frontend.service.jobs.settings import JobSettings


@dataclass(slots=True)
class ManagedJobRunnerProcess:
    """A private job-runner subprocess owned by the current parent process."""

    process: subprocess.Popen[str]


class ManagedJobRunnerProcessManager:
    """Start and stop the internal job-runner subprocess for frontend modes."""

    def __init__(self, *, settings_path: Path, job_settings: JobSettings) -> None:
        self._settings_path = settings_path
        self._job_settings = job_settings
        self._managed_process: ManagedJobRunnerProcess | None = None

    def start(self) -> None:
        if not self._job_settings.enabled:
            return
        if self._managed_process is not None:
            return

        process = self._spawn()
        self._wait_until_running(process)
        self._managed_process = ManagedJobRunnerProcess(process=process)

    def shutdown(self) -> None:
        if self._managed_process is None:
            return

        process = self._managed_process.process
        self._managed_process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _spawn(self) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vfront.frontend.job_runner",
                "--config",
                str(self._settings_path),
            ],
            cwd=str(self._settings_path.parent),
            text=True,
        )

    @staticmethod
    def _wait_until_running(process: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Internal job runner exited during startup.")
            time.sleep(0.1)
