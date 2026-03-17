"""Shared helpers for resolving frontend engine clients with health metadata."""

from __future__ import annotations

from dataclasses import dataclass

from vfront.backend_identity.fingerprint import fingerprint_from_health
from vfront.frontend.service.capabilities.service import BackendCapabilityService
from vfront.frontend.service.engine.router import EngineRouter, ResolvedEngineClient
from vfront.frontend.service.tool_parsing.service import ToolParsingService
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.model_registry import ResolvedModelTarget
from vfront.shared.engine.types import EngineHealth


@dataclass(slots=True, frozen=True)
class InitializedEngineClient:
    """Resolved engine client plus cached backend health metadata."""

    resolved: ResolvedEngineClient
    health: EngineHealth | None
    system_fingerprint: str | None

    @property
    def client(self) -> EngineClient:
        return self.resolved.client

    @property
    def target(self) -> ResolvedModelTarget:
        return self.resolved.target

    @property
    def adapter(self) -> str | None:
        return self.resolved.adapter


async def initialize_engine_client(
    *,
    resolved: ResolvedEngineClient,
    tool_parsing_service: ToolParsingService | None = None,
    backend_capability_service: BackendCapabilityService | None = None,
    require_tools: bool = False,
    health_by_backend: dict[str, EngineHealth] | None = None,
) -> InitializedEngineClient:
    """Fetch backend health once, validate tool parser needs, and derive fingerprint."""
    health = await _get_health(
        resolved=resolved,
        tool_parsing_service=tool_parsing_service,
        backend_capability_service=backend_capability_service,
        health_by_backend=health_by_backend,
    )
    if tool_parsing_service is not None:
        await tool_parsing_service.validate_target(
            target=resolved.target,
            client=resolved.client,
            require_tools=require_tools,
            health=health,
        )
    return InitializedEngineClient(
        resolved=resolved,
        health=health,
        system_fingerprint=(
            fingerprint_from_health(health, lora_name=resolved.adapter)
            if health is not None
            else None
        ),
    )


async def resolve_engine_client(
    *,
    model: str,
    engine_router: EngineRouter,
    tool_parsing_service: ToolParsingService | None = None,
    backend_capability_service: BackendCapabilityService | None = None,
    require_tools: bool = False,
    health_by_backend: dict[str, EngineHealth] | None = None,
) -> InitializedEngineClient:
    """Resolve a public model id and initialize its engine client metadata."""
    return await initialize_engine_client(
        resolved=engine_router.resolve_client(model),
        tool_parsing_service=tool_parsing_service,
        backend_capability_service=backend_capability_service,
        require_tools=require_tools,
        health_by_backend=health_by_backend,
    )


async def _get_health(
    *,
    resolved: ResolvedEngineClient,
    tool_parsing_service: ToolParsingService | None,
    backend_capability_service: BackendCapabilityService | None,
    health_by_backend: dict[str, EngineHealth] | None,
) -> EngineHealth | None:
    capability_service = backend_capability_service
    if capability_service is None and tool_parsing_service is not None:
        capability_service = getattr(tool_parsing_service, "backend_capability_service", None)
    backend = resolved.target.backend
    if capability_service is None and tool_parsing_service is None:
        return None
    if health_by_backend is not None:
        cached_health = health_by_backend.get(backend)
        if cached_health is not None:
            return cached_health
    if capability_service is not None:
        health = await capability_service.get_health(
            target=resolved.target,
            client=resolved.client,
        )
    else:
        assert tool_parsing_service is not None
        health = await tool_parsing_service.get_health(
            target=resolved.target,
            client=resolved.client,
        )
    if health_by_backend is not None:
        health_by_backend[backend] = health
    return health
