"""Dynamic access to the public model registry."""

from __future__ import annotations

from fastapiex.di import BaseService, Service
from fastapiex.settings import GetSettings

from vfront.shared.config.models import BackendConfig
from vfront.shared.engine.managed import managed_worker_backend
from vfront.shared.engine.model_registry import ModelRegistry, ModelSpec, ResolvedModelTarget


@Service("model_registry_service")
class ModelRegistryService(BaseService):
    """Resolve public model topology from the current settings snapshot."""

    @classmethod
    async def create(cls) -> ModelRegistryService:
        return cls()

    def _build_registry(self) -> ModelRegistry:
        routing_settings = GetSettings("frontend.routing")
        return ModelRegistry.from_settings(routing_settings)

    def _effective_backends(self) -> tuple[BackendConfig, ...]:
        routing_settings = GetSettings("frontend.routing")
        if routing_settings.transport != "managed":
            return tuple(routing_settings.backends)

        worker_settings = GetSettings("engine.managed_workers")
        return tuple(
            managed_worker_backend(
                worker_settings,
                backend_id=route.backend,
                runtime_model=route.runtime_model,
            )
            for route in routing_settings.runtime_routes
        )

    def get_registry(self) -> ModelRegistry:
        return self._build_registry()

    def resolve(self, model: str) -> ResolvedModelTarget | None:
        return self._build_registry().resolve(model)

    def get(self, model: str) -> ModelSpec | None:
        return self._build_registry().get(model)

    def list_all(self) -> tuple[ModelSpec, ...]:
        return self._build_registry().list_all()

    def list_backends(self) -> tuple[BackendConfig, ...]:
        return self._effective_backends()

    def get_backend(self, backend_id: str) -> BackendConfig | None:
        for backend in self._effective_backends():
            if backend.id == backend_id:
                return backend
        return None

    def list_backend_targets(self) -> tuple[ResolvedModelTarget, ...]:
        routing_settings = GetSettings("frontend.routing")
        targets: list[ResolvedModelTarget] = []
        for route in routing_settings.runtime_routes:
            targets.append(
                ResolvedModelTarget(
                    requested_model=route.name,
                    public_name=route.name,
                    backend=route.backend,
                    runtime_model=route.runtime_model,
                    adapter=route.adapter,
                    tool_parser=route.tool_parser,
                )
            )
        return tuple(targets)
