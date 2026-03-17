"""Engine-agnostic types for the EngineClient protocol.

These types form the boundary between the API layer and the inference engine.
Router and adapter code use these types, never engine-specific types directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from vfront.shared.engine.reasoning import ReasoningEffort


@dataclass(slots=True, frozen=True)
class ParsedToolCall:
    id: str | None = None
    type: str = "function"
    name: str = ""
    arguments: str = ""


@dataclass(slots=True, frozen=True)
class SemanticToolCallDelta:
    index: int
    id: str | None = None
    type: str | None = None
    name: str | None = None
    arguments_delta: str | None = None


@dataclass(slots=True, frozen=True)
class SemanticOutputDelta:
    content_delta: str | None = None
    tool_call_deltas: tuple[SemanticToolCallDelta, ...] = ()


@dataclass(slots=True, frozen=True)
class CompletionDelta:
    index: int
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str | None = None
    logprobs: tuple[dict[str, float], ...] | None = None
    cumulative_logprob: float | None = None
    reasoning_text: str | None = None
    semantic: SemanticOutputDelta | None = None
    parsed_tool_calls: tuple[ParsedToolCall, ...] | None = None


@dataclass(slots=True, frozen=True)
class GenerateOutput:
    request_id: str
    outputs: tuple[CompletionDelta, ...]
    prompt_token_ids: tuple[int, ...]
    finished: bool = False


@dataclass(slots=True, frozen=True)
class EmbeddingOutput:
    data: list[float]
    prompt_token_ids: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class LoRAAdapterInfo:
    name: str
    path: str
    id: int


InfillStrategy = Literal["native", "chat_fallback"]
EffectiveFIMMode = Literal["disabled", "native", "chat_fallback"]


@dataclass(slots=True, frozen=True)
class PreparedInfillPrompt:
    prompt_token_ids: tuple[int, ...]
    strategy: InfillStrategy
    internal_stop: tuple[str, ...] = ()
    response_open_tag: str | None = None
    response_close_tag: str | None = None
    supports_streaming: bool = True
    supports_logprobs: bool = True


@dataclass(slots=True, frozen=True)
class EngineHealth:
    ready: bool
    runtime_model: str
    max_model_len: int
    weights: str | None = None
    tokenizer_id: str | None = None
    quantization: str | None = None
    dtype: str | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    engine_version: str | None = None
    loaded_loras: tuple[str, ...] = ()
    transport: str = "unknown"
    available_tool_parsers: tuple[str, ...] = ()
    configured_tool_parser: str | None = None
    effective_tool_parser: str | None = None
    tool_parser_resolution_error: str | None = None
    available_reasoning_parsers: tuple[str, ...] = ()
    configured_reasoning_parser: str | None = None
    effective_reasoning_parser: str | None = None
    reasoning_parser_resolution_error: str | None = None
    configured_reasoning_effort_map: (
        dict[str, dict[str, bool | str | int | float | None] | None] | None
    ) = None
    configured_fim_policy: str | None = None
    effective_fim_mode: EffectiveFIMMode | None = None
    available_infill_strategies: tuple[InfillStrategy, ...] = ()
    configured_native_infill_profile: str | None = None
    configured_fallback_infill_strategy: str | None = None


@dataclass(slots=True)
class GenerateParams:
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    max_tokens: int | None = None
    stop: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    best_of: int | None = None
    json_schema: dict | None = None
    json_object: bool = False
    reasoning_effort: ReasoningEffort | None = None
    chat_template_applied: bool = False
    lora_name: str | None = None
    multimodal_data: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    tool_parser: str | None = None


@dataclass(slots=True)
class EncodeParams:
    dimensions: int | None = None
