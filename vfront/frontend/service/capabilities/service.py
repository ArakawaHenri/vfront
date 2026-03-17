"""Backend capability helpers shared across frontend validators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from fastapiex.di import BaseService, Service

from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.model_registry import ResolvedModelTarget
from vfront.shared.engine.types import EngineHealth

_DEFAULT_HEALTH_CACHE_TTL_SECONDS = 5.0


@dataclass(slots=True)
class _CachedHealth:
    health: EngineHealth
    expires_at: float


@Service("backend_capability_service")
class BackendCapabilityService(BaseService):
    def __init__(
        self,
        *,
        health_cache_ttl_seconds: float = _DEFAULT_HEALTH_CACHE_TTL_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._health_cache: dict[str, _CachedHealth] = {}
        self._health_cache_ttl_seconds = health_cache_ttl_seconds
        self._clock = clock or monotonic

    @classmethod
    async def create(cls) -> BackendCapabilityService:
        return cls()

    async def get_health(
        self,
        *,
        target: ResolvedModelTarget,
        client: EngineClient,
        refresh: bool = False,
    ) -> EngineHealth:
        now = self._clock()
        cached = self._health_cache.get(target.backend)
        if not refresh and cached is not None and cached.expires_at > now:
            return cached.health
        health = await client.health()
        self._health_cache[target.backend] = _CachedHealth(
            health=health,
            expires_at=now + self._health_cache_ttl_seconds,
        )
        return health

    @staticmethod
    def validate_backend_identity(
        *,
        target: ResolvedModelTarget,
        health: EngineHealth,
    ) -> None:
        if health.runtime_model != target.runtime_model:
            raise InvalidRequestError(
                f"Resolved backend '{target.backend}' is serving runtime model "
                f"'{health.runtime_model}', but frontend routing expected "
                f"'{target.runtime_model}'.",
                param="model",
            )
