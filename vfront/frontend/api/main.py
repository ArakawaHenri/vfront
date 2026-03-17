"""Root API router — health checks."""

from __future__ import annotations

from fastapi import APIRouter
from fastapiex.di import Inject
from fastapiex.settings import GetSettings
from starlette.responses import JSONResponse

from vfront.frontend.service.capabilities.service import BackendCapabilityService
from vfront.frontend.service.engine.registry_service import ModelRegistryService
from vfront.frontend.service.engine.router import EngineRouter

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@router.get("/ready")
async def ready(
    engine_router: EngineRouter = Inject("engine_router"),  # type: ignore[assignment]
    model_registry: ModelRegistryService = Inject("model_registry_service"),  # type: ignore[assignment]
    backend_capability_service: BackendCapabilityService = Inject(
        "backend_capability_service"
    ),  # type: ignore[assignment]
) -> JSONResponse:
    try:
        if not engine_router.has_loaded_engines():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "engine not loaded"},
            )

        routing_settings = GetSettings("frontend.routing")
        if routing_settings.transport == "local":
            return JSONResponse(content={"status": "ready", "models": engine_router.model_names})

        for target in model_registry.list_backend_targets():
            resolved = engine_router.client_for_target(target)
            health = await backend_capability_service.get_health(
                target=target,
                client=resolved.client,
                refresh=True,
            )
            backend_capability_service.validate_backend_identity(
                target=target,
                health=health,
            )
            if not health.ready:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "reason": (
                            f"backend '{target.backend}' for runtime model "
                            f"'{target.runtime_model}' is not ready"
                        ),
                    },
                )

        return JSONResponse(content={"status": "ready", "models": engine_router.model_names})
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(e)},
        )
