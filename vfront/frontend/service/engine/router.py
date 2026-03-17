"""Runtime model router that resolves public IDs to engine clients."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapiex.settings import GetSettings

from vfront.frontend.middleware.exceptions import NotFoundError
from vfront.frontend.service.engine.registry_service import ModelRegistryService
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.http_client import RemoteHttpEngineClient
from vfront.shared.engine.model_registry import ResolvedModelTarget

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ResolvedEngineClient:
    """A resolved public model target plus its bound engine client."""

    target: ResolvedModelTarget
    client: EngineClient

    @property
    def adapter(self) -> str | None:
        return self.target.adapter


class EngineRouter:
    """Resolve public model IDs to concrete engine clients."""

    def __init__(
        self,
        backend: Any | None,
        model_registry: ModelRegistryService,
        remote_client_factory: Callable[..., EngineClient] = RemoteHttpEngineClient,
    ) -> None:
        self._backend = backend
        self._model_registry = model_registry
        self._remote_client_factory = remote_client_factory
        self._remote_clients: dict[str, EngineClient] = {}

    async def close(self) -> None:
        errors: list[Exception] = []
        remote_clients = list(self._remote_clients.items())
        self._remote_clients.clear()
        for backend, client in remote_clients:
            close = getattr(client, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Failed to close remote engine client for engine '%s'.",
                    backend,
                )
                errors.append(exc)
        if errors:
            raise ExceptionGroup(
                "Failed to close one or more remote engine clients.",
                errors,
            )

    def resolve_target(self, model: str) -> ResolvedModelTarget:
        target = self._model_registry.resolve(model)
        if target is None:
            raise NotFoundError(f"The model '{model}' does not exist.")
        return target

    def resolve_client(self, model: str) -> ResolvedEngineClient:
        target = self.resolve_target(model)
        return self.client_for_target(target)

    def client_for_target(self, target: ResolvedModelTarget) -> ResolvedEngineClient:
        return ResolvedEngineClient(
            target=target,
            client=self._client_for_target(target),
        )

    def has_loaded_engines(self) -> bool:
        routing_settings = GetSettings("frontend.routing")
        if routing_settings.transport != "local":
            return bool(self._model_registry.list_backends())
        if self._backend is None:
            return False
        return self._backend.has_loaded_engines()

    @property
    def model_names(self) -> list[str]:
        routing_settings = GetSettings("frontend.routing")
        if routing_settings.transport != "local":
            return [backend.id for backend in self._model_registry.list_backends()]
        if self._backend is None:
            return []
        return self._backend.runtime_models

    def _client_for_target(self, target: ResolvedModelTarget) -> EngineClient:
        routing_settings = GetSettings("frontend.routing")
        if routing_settings.transport == "local":
            if self._backend is None:
                raise RuntimeError("Local engine backend is not installed in this process.")
            return self._backend.for_runtime_model(target.runtime_model)

        backend_target = self._model_registry.get_backend(target.backend)
        if backend_target is None:
            raise NotFoundError(f"No backend is configured for backend '{target.backend}'.")
        client = self._remote_clients.get(target.backend)
        if client is None:
            client = self._remote_client_factory(
                base_url=backend_target.base_url,
                api_key=backend_target.api_key,
                timeout_seconds=backend_target.timeout_seconds,
            )
            self._remote_clients[target.backend] = client
        return client
