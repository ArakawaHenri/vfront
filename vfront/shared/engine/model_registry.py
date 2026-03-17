"""Model registry for frontend-declared public model routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


class _HasRoutingModel(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def runtime_model(self) -> str: ...

    @property
    def adapter(self) -> str | None: ...

    @property
    def capabilities(self) -> list[str]: ...

    @property
    def tool_parser(self) -> str | None: ...


class _HasRoutingModels(Protocol):
    @property
    def models(self) -> list[_HasRoutingModel]: ...


@dataclass(slots=True, frozen=True)
class BaseModelSpec:
    kind: Literal["base"] = "base"
    name: str = ""
    backend: str = ""
    runtime_model: str = ""
    capabilities: tuple[str, ...] = ("text",)
    tool_parser: str | None = None


@dataclass(slots=True, frozen=True)
class LoRAModelSpec:
    kind: Literal["lora"] = "lora"
    name: str = ""
    backend: str = ""
    runtime_model: str = ""
    adapter: str = ""
    capabilities: tuple[str, ...] = ("text",)
    tool_parser: str | None = None


ModelSpec = BaseModelSpec | LoRAModelSpec


@dataclass(slots=True, frozen=True)
class ResolvedModelTarget:
    requested_model: str
    public_name: str
    backend: str
    runtime_model: str
    adapter: str | None = None
    tool_parser: str | None = None


@dataclass(slots=True, frozen=True)
class ModelRegistry:
    base_models: tuple[BaseModelSpec, ...]
    lora_models: tuple[LoRAModelSpec, ...]

    @classmethod
    def from_settings(cls, routing_settings: _HasRoutingModels) -> ModelRegistry:
        base_models: list[BaseModelSpec] = []
        lora_models: list[LoRAModelSpec] = []
        for model in routing_settings.models:
            if model.adapter is None:
                base_models.append(
                    BaseModelSpec(
                        name=model.name,
                        backend=model.backend,
                        runtime_model=model.runtime_model,
                        capabilities=tuple(model.capabilities),
                        tool_parser=model.tool_parser,
                    )
                )
                continue
            lora_models.append(
                LoRAModelSpec(
                    name=model.name,
                    backend=model.backend,
                    runtime_model=model.runtime_model,
                    adapter=model.adapter,
                    capabilities=tuple(model.capabilities),
                    tool_parser=model.tool_parser,
                )
            )

        return cls(
            base_models=tuple(base_models),
            lora_models=tuple(lora_models),
        )

    def get(self, name: str) -> ModelSpec | None:
        for base_spec in self.base_models:
            if base_spec.name == name:
                return base_spec
        for lora_spec in self.lora_models:
            if lora_spec.name == name:
                return lora_spec
        return None

    def list_all(self) -> tuple[ModelSpec, ...]:
        return self.base_models + self.lora_models

    def resolve(self, name: str) -> ResolvedModelTarget | None:
        spec = self.get(name)
        if spec is None:
            return None
        if isinstance(spec, LoRAModelSpec):
            return ResolvedModelTarget(
                requested_model=name,
                public_name=spec.name,
                backend=spec.backend,
                runtime_model=spec.runtime_model,
                adapter=spec.adapter,
                tool_parser=spec.tool_parser,
            )
        return ResolvedModelTarget(
            requested_model=name,
            public_name=spec.name,
            backend=spec.backend,
            runtime_model=spec.runtime_model,
            adapter=None,
            tool_parser=spec.tool_parser,
        )
