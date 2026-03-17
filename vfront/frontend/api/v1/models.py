"""/v1/models — Models CRUD endpoint."""

from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter
from fastapiex.di import Inject

from vfront.frontend.middleware.exceptions import InvalidRequestError, NotFoundError
from vfront.frontend.service.engine.registry_service import ModelRegistryService
from vfront.frontend.service.engine.router import EngineRouter
from vfront.protocol.model import ModelDeleted, ModelList, ModelObject

router = APIRouter()
logger = logging.getLogger(__name__)

_SYNTHETIC_MODEL_CREATED_EPOCH = 1_704_067_200


def _stable_model_created(*, model_id: str, backend: str, kind: str) -> int:
    payload = f"{kind}:{backend}:{model_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return _SYNTHETIC_MODEL_CREATED_EPOCH + (
        int.from_bytes(digest[:4], "big") % 31_536_000
    )


def _build_model_object(
    *,
    model_id: str,
    backend: str,
    kind: str,
    capabilities: tuple[str, ...] | list[str],
    context_window: int | None,
) -> ModelObject:
    return ModelObject(
        id=model_id,
        created=_stable_model_created(model_id=model_id, backend=backend, kind=kind),
        owned_by="lora" if kind == "lora" else "system",
        context_window=context_window,
        capabilities=list(capabilities),
    )


async def _resolve_context_window(
    *,
    model_id: str,
    backend: str,
    target,
    engine_router: EngineRouter,
) -> int | None:
    if target is None:
        return None

    try:
        resolved = engine_router.client_for_target(target)
    except Exception as exc:
        logger.warning(
            "Failed to resolve context window for model '%s' on backend '%s': %s",
            model_id,
            backend,
            exc,
        )
        return None

    try:
        health = await resolved.client.health()
    except Exception as exc:
        logger.warning(
            "Failed to fetch engine health for model '%s' on backend '%s': %s",
            model_id,
            backend,
            exc,
        )
        max_len = resolved.client.max_model_len
    else:
        max_len = health.max_model_len
    return max_len if max_len > 0 else None


@router.get("/models", response_model=ModelList)
async def list_models(
    model_registry: ModelRegistryService = Inject("model_registry_service"),
    engine_router: EngineRouter = Inject("engine_router"),
):
    registry = model_registry.get_registry()

    context_by_backend: dict[str, int | None] = {}

    models: list[ModelObject] = []
    for spec in registry.list_all():
        if spec.backend not in context_by_backend:
            context_by_backend[spec.backend] = await _resolve_context_window(
                model_id=spec.name,
                backend=spec.backend,
                target=registry.resolve(spec.name),
                engine_router=engine_router,
            )

        models.append(
            _build_model_object(
                model_id=spec.name,
                backend=spec.backend,
                kind=spec.kind,
                capabilities=spec.capabilities,
                context_window=context_by_backend.get(spec.backend),
            )
        )
    return ModelList(data=models)


@router.get("/models/{model_id:path}", response_model=ModelObject)
async def get_model(
    model_id: str,
    model_registry: ModelRegistryService = Inject("model_registry_service"),
    engine_router: EngineRouter = Inject("engine_router"),
):
    registry = model_registry.get_registry()
    spec = registry.get(model_id)
    if spec is None:
        raise NotFoundError(f"The model '{model_id}' does not exist.")

    context_window = await _resolve_context_window(
        model_id=model_id,
        backend=spec.backend,
        target=registry.resolve(model_id),
        engine_router=engine_router,
    )

    return _build_model_object(
        model_id=model_id,
        backend=spec.backend,
        kind=spec.kind,
        capabilities=spec.capabilities,
        context_window=context_window,
    )


@router.delete("/models/{model_id:path}", response_model=ModelDeleted)
async def delete_model(
    model_id: str,
    model_registry: ModelRegistryService = Inject("model_registry_service"),
):
    spec = model_registry.get(model_id)
    if spec is None:
        raise NotFoundError(f"The model '{model_id}' does not exist.")
    raise InvalidRequestError(
        "Runtime model catalog mutation is not supported. "
        "Remove the route from `frontend.routing.models` and restart the service.",
        param="model",
    )
