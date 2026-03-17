"""FastAPI application factory.

Initialization pipeline following neo_tx_classifier pattern:
1. Ensure settings are initialized (idempotent)
2. Setup logging
3. Create FastAPI app with lifespan
4. Add CORS middleware
5. Register exception handlers
6. Add request logging middleware
7. Install DI (scans vfront.service for @Service classes)
8. Include routers
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapiex.di import install_di
from fastapiex.settings import GetSettings
from starlette.exceptions import HTTPException

from vfront.core.logging import setup_logging, shutdown_logging
from vfront.frontend.api.main import router as health_router
from vfront.frontend.api.v1.main import router as v1_router
from vfront.frontend.middleware.auth import ApiKeyAuthMiddleware
from vfront.frontend.middleware.exceptions import (
    OpenAIError,
    engine_input_error_handler,
    global_exception_handler,
    http_exception_handler,
    openai_error_handler,
    validation_error_handler,
)
from vfront.frontend.middleware.logging import RequestLoggingMiddleware
from vfront.shared.engine.errors import EngineInputError


async def _cancel_background_tasks() -> None:
    """Background jobs are owned by JobRunnerService and cleaned up by DI."""
    await asyncio.sleep(0)


def create_application() -> FastAPI:
    # 2. Setup logging
    app_settings = GetSettings("frontend.app")
    setup_logging(log_dir=app_settings.log_dir, debug=app_settings.debug_mode)

    # 3. Create FastAPI with lifespan
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        import logging

        _logger = logging.getLogger("vfront")
        routing_settings = GetSettings("frontend.routing")
        _logger.info(
            "Engines ready. Serving public model routes: %s",
            [
                {
                    "name": model.name,
                    "backend": model.backend,
                    "runtime_model": model.runtime_model,
                    "adapter": model.adapter,
                    "tool_parser": model.tool_parser,
                }
                for model in routing_settings.models
            ],
        )
        yield
        _logger.info("Shutting down...")
        # Cancel all in-flight background tasks
        await _cancel_background_tasks()
        await shutdown_logging()

    app = FastAPI(
        title=app_settings.title,
        version=app_settings.version,
        description=app_settings.description,
        lifespan=lifespan,
    )

    # 4. Add CORS
    cors_origins = list(app_settings.cors_origins)
    if cors_origins:
        allow_credentials = "*" not in cors_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 5. Register exception handlers
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(EngineInputError, engine_input_error_handler)
    app.add_exception_handler(OpenAIError, openai_error_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # 6. Add API key auth middleware for `/v1/*` routes when configured.
    app.add_middleware(ApiKeyAuthMiddleware)

    # 7. Add request logging middleware.
    #    Added after auth so it becomes the outer middleware and still logs auth failures.
    app.add_middleware(RequestLoggingMiddleware)

    # 8. Install DI — scans vfront.service for @Service classes (engine backend/router/etc.)
    #    eager=True services are created during startup, destroyed on shutdown
    routing_settings = GetSettings("frontend.routing")
    service_packages = ["vfront.frontend.service", "vfront.frontend.remote_service"]
    if routing_settings.transport == "local":
        service_packages = [
            "vfront.frontend.service",
            "vfront.frontend.embedded_service",
            "vfront.engine.service",
        ]
    install_di(app, service_packages=service_packages)

    # 9. Include routers
    app.include_router(health_router)
    app.include_router(v1_router)

    return app
