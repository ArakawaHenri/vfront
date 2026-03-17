"""Engine runtime settings."""

from __future__ import annotations

from fastapiex.settings import BaseSettings, Settings
from pydantic import ConfigDict, Field, model_validator

from vfront.shared.config.models import RuntimeModelConfig


@Settings("engine.runtime")
class RuntimeSettings(BaseSettings):
    model_config = ConfigDict(extra="forbid")

    fim: bool | None = None
    models: list[RuntimeModelConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_models(self) -> RuntimeSettings:
        runtime_ids = [model.id for model in self.models]
        if len(set(runtime_ids)) != len(runtime_ids):
            raise ValueError("engine.runtime.models[].id values must be unique.")
        return self

    @property
    def primary_runtime_model(self) -> RuntimeModelConfig:
        return self.models[0]
