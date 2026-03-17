"""Reusable adapter configuration fragments."""

from __future__ import annotations

from fastapiex.settings import BaseSettings
from pydantic import Field, model_validator


class AdapterModuleConfig(BaseSettings):
    id: str
    path: str
    runtime_model: str


class AdapterSettings(BaseSettings):
    enabled: bool = False
    modules: list[AdapterModuleConfig] = Field(default_factory=list)
    max_rank: int = 16
    max_loras: int = 1
    max_cpu_loras: int | None = None

    @model_validator(mode="after")
    def _validate_modules(self) -> AdapterSettings:
        seen: set[tuple[str, str]] = set()
        for module in self.modules:
            key = (module.runtime_model, module.id)
            if key in seen:
                raise ValueError(
                    "adapter module ids must be unique per runtime_model; "
                    f"duplicate ({module.runtime_model}, {module.id})"
                )
            seen.add(key)
        return self
