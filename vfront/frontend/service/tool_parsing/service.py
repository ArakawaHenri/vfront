"""Frontend-side parser capability service.

This service does not run parsers. It validates that configured parser names are
known to the installed vLLM package and exposes the available parser list to the
compatibility layer. Actual parsing is performed in the engine runtime where the
tokenizer lives.
"""

from __future__ import annotations

from time import monotonic

from fastapiex.di import BaseService, Require, Service

from vfront.engine.service.tool_parsing import list_available_tool_parsers
from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.frontend.service.capabilities.service import BackendCapabilityService
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.model_registry import ResolvedModelTarget
from vfront.shared.engine.types import EngineHealth


@Service("tool_parsing_service")
class ToolParsingService(BaseService):
    def __init__(
        self,
        *,
        backend_capability_service: BackendCapabilityService | None = None,
        health_cache_ttl_seconds: float | None = None,
    ) -> None:
        if backend_capability_service is not None:
            self._backend_capability_service = backend_capability_service
            return

        if health_cache_ttl_seconds is not None:
            self._backend_capability_service = BackendCapabilityService(
                health_cache_ttl_seconds=health_cache_ttl_seconds,
                clock=lambda: monotonic(),
            )
            return

        self._backend_capability_service = BackendCapabilityService(clock=lambda: monotonic())

    @classmethod
    async def create(
        cls,
        backend_capability_service: BackendCapabilityService = Require(
            "backend_capability_service"
        ),  # type: ignore[assignment]
    ) -> ToolParsingService:
        return cls(backend_capability_service=backend_capability_service)

    def list_available_parsers(self, health: EngineHealth | None = None) -> list[str]:
        if health is not None:
            return list(health.available_tool_parsers)
        return sorted(list_available_tool_parsers())

    @property
    def backend_capability_service(self) -> BackendCapabilityService:
        return self._backend_capability_service

    async def get_health(
        self,
        *,
        target: ResolvedModelTarget,
        client: EngineClient,
        refresh: bool = False,
    ) -> EngineHealth:
        return await self._backend_capability_service.get_health(
            target=target,
            client=client,
            refresh=refresh,
        )

    async def validate_target(
        self,
        *,
        target: ResolvedModelTarget,
        client: EngineClient,
        require_tools: bool,
        health: EngineHealth | None = None,
    ) -> None:
        if health is None:
            health = await self.get_health(target=target, client=client)
        self._backend_capability_service.validate_backend_identity(
            target=target,
            health=health,
        )
        if not require_tools:
            return
        if target.tool_parser is None:
            import logging

            logging.getLogger(__name__).warning(
                "Tool parser required but missing on resolved target: "
                "requested_model=%s public_name=%s backend=%s runtime_model=%s adapter=%s",
                target.requested_model,
                target.public_name,
                target.backend,
                target.runtime_model,
                target.adapter,
            )
            raise InvalidRequestError(
                "Tool parsing is disabled for this model. "
                f"Resolved target: model='{target.public_name}', backend='{target.backend}', "
                f"runtime_model='{target.runtime_model}', adapter='{target.adapter}'. "
                "Configure `frontend.routing.models[].tool_parser` to an explicit parser name.",
                param="model",
            )
        if target.tool_parser == "auto":
            raise InvalidRequestError(
                "The 'auto' tool parser resolution is no longer supported. "
                "Please explicitly configure a specific tool parser. "
                f"Available parsers: {self.list_available_parsers(health)}",
                param="model",
            )
        if target.tool_parser not in health.available_tool_parsers:
            raise InvalidRequestError(
                f"Configured tool parser '{target.tool_parser}' is not available on "
                f"backend '{target.backend}'. Available parsers: "
                f"{self.list_available_parsers(health)}",
                param="model",
            )
        if health.tool_parser_resolution_error:
            raise InvalidRequestError(
                f"Configured tool parser '{target.tool_parser}' is invalid for model "
                f"'{target.public_name}': {health.tool_parser_resolution_error}",
                param="model",
            )
