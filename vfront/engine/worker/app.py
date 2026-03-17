"""ASGI app for the internal engine worker."""

from __future__ import annotations

from vfront.engine.worker.factory import create_engine_worker_application

app = create_engine_worker_application()
