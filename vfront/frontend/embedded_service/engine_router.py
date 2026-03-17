"""Embedded-local engine-router DI registration."""

from __future__ import annotations

from fastapiex.di import BaseService, Require, Service

from vfront.engine.service.local import LocalEngineManager
from vfront.frontend.service.engine.registry_service import ModelRegistryService
from vfront.frontend.service.engine.router import EngineRouter


@Service("engine_router")
class EmbeddedEngineRouterService(BaseService):
    @classmethod
    async def create(
        cls,
        backend: LocalEngineManager = Require("engine_backend"),  # type: ignore[assignment]
        model_registry: ModelRegistryService = Require("model_registry_service"),  # type: ignore[assignment]
    ) -> EngineRouter:
        return EngineRouter(backend=backend, model_registry=model_registry)

    @classmethod
    async def destroy(cls, instance: EngineRouter) -> None:
        await instance.close()
