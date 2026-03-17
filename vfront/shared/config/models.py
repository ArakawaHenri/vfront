"""Reusable schema fragments for routing, runtime model topology, and infill."""

from __future__ import annotations

from typing import Literal

from fastapiex.settings import BaseSettings
from pydantic import Field, model_validator

from vfront.shared.engine.reasoning import ReasoningEffortKey, format_reasoning_effort_key

ReasoningTemplateValue = bool | str | int | float | None
ReasoningTemplateKwargs = dict[str, ReasoningTemplateValue]
ReasoningPolicyValue = ReasoningTemplateKwargs | None


class ReasoningConfig(BaseSettings):
    """Model-specific reasoning override map.

    `effort_map` is the highest-priority source of truth. Keys may be one of
    the OpenAI reasoning efforts or `null` / `~` to represent the implicit
    request state where no `reasoning_effort` was provided.

    If a key is present, its kwargs are used as-is; an empty dict explicitly
    suppresses any auto-detected fallback for that key. Without an explicit
    per-effort entry, vfront only auto-passes `reasoning_effort` for templates
    it can conservatively prove to consume that kwarg directly; bridges such as
    `enable_thinking` must be configured explicitly.
    """

    effort_map: dict[ReasoningEffortKey, ReasoningPolicyValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_effort_map(self) -> ReasoningConfig:
        for effort, kwargs in self.effort_map.items():
            if kwargs is None:
                continue
            if not isinstance(kwargs, dict):
                raise ValueError(
                    "reasoning.effort_map."
                    f"{format_reasoning_effort_key(effort)} must be an object "
                    "of template kwargs or null."
                )
            for key in kwargs:
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        "reasoning.effort_map."
                        f"{format_reasoning_effort_key(effort)} contains an invalid "
                        "kwarg name."
                    )
        return self


InfillNativeProfile = Literal["starcoder", "custom"]
InfillFallbackStrategy = Literal["chat_tagged"]


class NativeInfillConfig(BaseSettings):
    """Runtime-native fill-in-middle configuration."""

    profile: InfillNativeProfile = "custom"
    prefix_token: str | None = None
    suffix_token: str | None = None
    middle_token: str | None = None
    template: str | None = None
    extra_stop: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_native_profile(self) -> NativeInfillConfig:
        if self.profile == "starcoder":
            if any(
                value is not None
                for value in (
                    self.prefix_token,
                    self.suffix_token,
                    self.middle_token,
                    self.template,
                )
            ):
                raise ValueError(
                    "engine.runtime.models[].infill.native with profile='starcoder' "
                    "must not override custom token/template fields."
                )
            return self

        required = {
            "prefix_token": self.prefix_token,
            "suffix_token": self.suffix_token,
            "middle_token": self.middle_token,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "engine.runtime.models[].infill.native profile='custom' requires "
                + ", ".join(missing)
                + "."
            )
        return self


class TaggedChatFallbackInfillConfig(BaseSettings):
    """Best-effort infill fallback via a tagged chat-template rewrite."""

    strategy: InfillFallbackStrategy = "chat_tagged"
    open_tag: str = "<INFILL>"
    close_tag: str = "</INFILL>"
    system_prompt: str = (
        "You are a fill-in-the-middle engine. Return only the missing middle text "
        "between PREFIX and SUFFIX. Emit exactly one tagged block and no other text."
    )
    user_template: str = (
        "PREFIX:\n{{ prefix }}\n\n"
        "SUFFIX:\n{{ suffix }}\n\n"
        "Return exactly one block: {{ open_tag }}...{{ close_tag }}"
    )

    @model_validator(mode="after")
    def _validate_tags(self) -> TaggedChatFallbackInfillConfig:
        if not self.open_tag or not self.close_tag:
            raise ValueError(
                "engine.runtime.models[].infill.fallback requires non-empty open_tag "
                "and close_tag."
            )
        if self.open_tag == self.close_tag:
            raise ValueError(
                "engine.runtime.models[].infill.fallback open_tag and close_tag "
                "must differ."
            )
        return self


class InfillConfig(BaseSettings):
    """Runtime infill support: native FIM and/or best-effort fallback."""

    native: NativeInfillConfig | None = None
    fallback: TaggedChatFallbackInfillConfig | None = None

    @model_validator(mode="after")
    def _validate_presence(self) -> InfillConfig:
        if self.native is None and self.fallback is None:
            raise ValueError(
                "engine.runtime.models[].infill must configure at least one of "
                "'native' or 'fallback'."
            )
        return self


class RuntimeModelConfig(BaseSettings):
    """A single engine runtime model instance."""

    id: str
    weights: str
    tokenizer: str | None = None
    max_model_len: int | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["text"])
    tool_parser: str | None = None
    reasoning: ReasoningConfig | None = None
    reasoning_parser: str | None = None
    fim: bool | None = None
    infill: InfillConfig | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    trust_remote_code: bool = False
    quantization: str | None = None


class BackendConfig(BaseSettings):
    id: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 300.0
