"""Frontend routing settings for engine backends."""

from __future__ import annotations

from typing import Literal

from fastapiex.settings import BaseSettings, Settings
from pydantic import ConfigDict, Field, model_validator

from vfront.shared.config.models import BackendConfig


class RoutingModelConfig(BaseSettings):
    model_config = ConfigDict(extra="forbid")

    name: str
    backend: str
    runtime_model: str
    adapter: str | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["text"])
    tool_parser: str | None = None


@Settings("frontend.routing")
class RoutingSettings(BaseSettings):
    model_config = ConfigDict(extra="forbid")

    transport: Literal["local", "managed", "remote"] = "local"
    batch_concurrency: int = 64
    models: list[RoutingModelConfig] = Field(default_factory=list)
    backends: list[BackendConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_models(self) -> RoutingSettings:
        names = [model.name for model in self.models]
        if len(set(names)) != len(names):
            raise ValueError("frontend.routing.models[].name values must be unique.")

        backend_ids = [backend.id for backend in self.backends]
        if len(set(backend_ids)) != len(backend_ids):
            raise ValueError("frontend.routing.backends[].id values must be unique.")

        runtime_to_backend: dict[str, str] = {}
        backend_to_runtime: dict[str, str] = {}
        for model in self.models:
            known_backend = runtime_to_backend.get(model.runtime_model)
            if known_backend is not None and known_backend != model.backend:
                raise ValueError(
                    f"frontend routing runtime_model '{model.runtime_model}' is mapped to "
                    f"multiple backends: '{known_backend}' and '{model.backend}'."
                )
            runtime_to_backend[model.runtime_model] = model.backend

            if self.transport == "remote":
                known_runtime = backend_to_runtime.get(model.backend)
                if known_runtime is not None and known_runtime != model.runtime_model:
                    raise ValueError(
                        f"frontend routing backend '{model.backend}' is mapped to multiple "
                        f"runtime models: '{known_runtime}' and '{model.runtime_model}'."
                    )
                backend_to_runtime[model.backend] = model.runtime_model

        if self.transport == "remote":
            if not self.backends:
                raise ValueError(
                    "frontend.routing.backends must contain at least one backend in remote mode."
                )
            known_backend_ids = set(backend_ids)
            missing = sorted({model.backend for model in self.models} - known_backend_ids)
            if missing:
                raise ValueError(
                    "frontend.routing.models reference unknown backends: "
                    + ", ".join(missing)
                )
        elif self.backends:
            known_backend_ids = set(backend_ids)
            missing = sorted({model.backend for model in self.models} - known_backend_ids)
            if missing:
                raise ValueError(
                    "frontend.routing.models reference unknown backends: "
                    + ", ".join(missing)
                )

        return self

    @property
    def base_routes(self) -> list[RoutingModelConfig]:
        return [model for model in self.models if model.adapter is None]

    @property
    def runtime_models(self) -> list[str]:
        ordered: list[str] = []
        for model in self.models:
            if model.runtime_model not in ordered:
                ordered.append(model.runtime_model)
        return ordered

    @property
    def runtime_routes(self) -> list[RoutingModelConfig]:
        routes: list[RoutingModelConfig] = []
        seen_runtime_models: set[str] = set()
        for model in self.models:
            if model.runtime_model in seen_runtime_models:
                continue
            routes.append(model)
            seen_runtime_models.add(model.runtime_model)
        return routes
