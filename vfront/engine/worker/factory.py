"""Factory for the internal engine-worker ASGI application."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapiex.di import install_di
from fastapiex.settings import GetSettings
from starlette.responses import Response

from vfront.core.logging import setup_logging, shutdown_logging
from vfront.engine.worker.routes import router as engine_worker_router
from vfront.shared.engine.errors import EngineInputError


def create_engine_worker_application() -> FastAPI:
    app_settings = GetSettings("engine.app")
    setup_logging(log_dir=app_settings.log_dir, debug=app_settings.debug_mode)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        del app
        yield
        await shutdown_logging()

    app = FastAPI(
        title=f"{app_settings.title} Engine Worker",
        version=app_settings.version,
        description="Internal engine worker API",
        lifespan=lifespan,
    )

    async def engine_input_error_handler(
        request: Request, exc: EngineInputError
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=400, content={"detail": exc.message})

    app.add_exception_handler(
        EngineInputError,
        cast(
            "Callable[[Request, Exception], Response | Awaitable[Response]]",
            engine_input_error_handler,
        ),
    )
    install_di(app, service_packages=["vfront.engine.service"])
    app.include_router(engine_worker_router)
    return app
