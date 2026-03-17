"""Remote/managed engine-router DI registration."""

from __future__ import annotations

from fastapiex.di import BaseService, Require, Service

from vfront.frontend.service.engine.registry_service import ModelRegistryService
from vfront.frontend.service.engine.router import EngineRouter


@Service("engine_router")
class RemoteEngineRouterService(BaseService):
    @classmethod
    async def create(
        cls,
        model_registry: ModelRegistryService = Require("model_registry_service"),  # type: ignore[assignment]
    ) -> EngineRouter:
        return EngineRouter(backend=None, model_registry=model_registry)

    @classmethod
    async def destroy(cls, instance: EngineRouter) -> None:
        await instance.close()
