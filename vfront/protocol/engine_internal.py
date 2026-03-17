"""Internal protocol between control-plane and engine workers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from vfront.shared.engine.reasoning import ReasoningEffort
from vfront.shared.engine.types import (
    CompletionDelta,
    EffectiveFIMMode,
    EmbeddingOutput,
    EncodeParams,
    EngineHealth,
    GenerateOutput,
    GenerateParams,
    InfillStrategy,
    LoRAAdapterInfo,
    ParsedToolCall,
    PreparedInfillPrompt,
    SemanticOutputDelta,
    SemanticToolCallDelta,
)


class InternalCompletionDelta(BaseModel):
    index: int
    text: str
    token_ids: list[int]
    finish_reason: str | None = None
    logprobs: list[dict[str, float]] | None = None
    cumulative_logprob: float | None = None
    reasoning_text: str | None = None
    semantic: InternalSemanticOutputDelta | None = None
    parsed_tool_calls: list[InternalParsedToolCall] | None = None

    @classmethod
    def from_engine(cls, delta: CompletionDelta) -> InternalCompletionDelta:
        return cls(
            index=delta.index,
            text=delta.text,
            token_ids=list(delta.token_ids),
            finish_reason=delta.finish_reason,
            logprobs=list(delta.logprobs) if delta.logprobs is not None else None,
            cumulative_logprob=delta.cumulative_logprob,
            reasoning_text=delta.reasoning_text,
            semantic=(
                InternalSemanticOutputDelta.from_engine(delta.semantic)
                if delta.semantic is not None
                else None
            ),
            parsed_tool_calls=(
                [InternalParsedToolCall.from_engine(item) for item in delta.parsed_tool_calls]
                if delta.parsed_tool_calls is not None
                else None
            ),
        )

    def to_engine(self) -> CompletionDelta:
        return CompletionDelta(
            index=self.index,
            text=self.text,
            token_ids=tuple(self.token_ids),
            finish_reason=self.finish_reason,
            logprobs=tuple(self.logprobs) if self.logprobs is not None else None,
            cumulative_logprob=self.cumulative_logprob,
            reasoning_text=self.reasoning_text,
            semantic=self.semantic.to_engine() if self.semantic is not None else None,
            parsed_tool_calls=(
                tuple(item.to_engine() for item in self.parsed_tool_calls)
                if self.parsed_tool_calls is not None
                else None
            ),
        )


class InternalParsedToolCall(BaseModel):
    id: str | None = None
    type: str = "function"
    name: str = ""
    arguments: str = ""

    @classmethod
    def from_engine(cls, tool_call: ParsedToolCall) -> InternalParsedToolCall:
        return cls(
            id=tool_call.id,
            type=tool_call.type,
            name=tool_call.name,
            arguments=tool_call.arguments,
        )

    def to_engine(self) -> ParsedToolCall:
        return ParsedToolCall(
            id=self.id,
            type=self.type,
            name=self.name,
            arguments=self.arguments,
        )


class InternalSemanticToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    type: str | None = None
    name: str | None = None
    arguments_delta: str | None = None

    @classmethod
    def from_engine(cls, delta: SemanticToolCallDelta) -> InternalSemanticToolCallDelta:
        return cls(
            index=delta.index,
            id=delta.id,
            type=delta.type,
            name=delta.name,
            arguments_delta=delta.arguments_delta,
        )

    def to_engine(self) -> SemanticToolCallDelta:
        return SemanticToolCallDelta(
            index=self.index,
            id=self.id,
            type=self.type,
            name=self.name,
            arguments_delta=self.arguments_delta,
        )


class InternalSemanticOutputDelta(BaseModel):
    content_delta: str | None = None
    tool_call_deltas: list[InternalSemanticToolCallDelta] = Field(default_factory=list)

    @classmethod
    def from_engine(cls, delta: SemanticOutputDelta) -> InternalSemanticOutputDelta:
        return cls(
            content_delta=delta.content_delta,
            tool_call_deltas=[
                InternalSemanticToolCallDelta.from_engine(item)
                for item in delta.tool_call_deltas
            ],
        )

    def to_engine(self) -> SemanticOutputDelta:
        return SemanticOutputDelta(
            content_delta=self.content_delta,
            tool_call_deltas=tuple(item.to_engine() for item in self.tool_call_deltas),
        )


class InternalGenerateParams(BaseModel):
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

    @classmethod
    def from_engine(cls, params: GenerateParams) -> InternalGenerateParams:
        return cls(
            temperature=params.temperature,
            top_p=params.top_p,
            n=params.n,
            max_tokens=params.max_tokens,
            stop=params.stop,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            logprobs=params.logprobs,
            prompt_logprobs=params.prompt_logprobs,
            seed=params.seed,
            logit_bias=params.logit_bias,
            best_of=params.best_of,
            json_schema=params.json_schema,
            json_object=params.json_object,
            reasoning_effort=params.reasoning_effort,
            chat_template_applied=params.chat_template_applied,
            lora_name=params.lora_name,
            multimodal_data=params.multimodal_data,
            tools=params.tools,
            tool_choice=params.tool_choice,
            tool_parser=params.tool_parser,
        )

    def to_engine(self) -> GenerateParams:
        return GenerateParams(**self.model_dump())


class InternalEncodeParams(BaseModel):
    dimensions: int | None = None

    @classmethod
    def from_engine(cls, params: EncodeParams) -> InternalEncodeParams:
        return cls(dimensions=params.dimensions)

    def to_engine(self) -> EncodeParams:
        return EncodeParams(dimensions=self.dimensions)


class GenerateRequest(BaseModel):
    prompt_token_ids: list[int]
    params: InternalGenerateParams
    request_id: str


class GenerateOutputChunk(BaseModel):
    request_id: str
    outputs: list[InternalCompletionDelta]
    prompt_token_ids: list[int]
    finished: bool = False

    @classmethod
    def from_engine(cls, output: GenerateOutput) -> GenerateOutputChunk:
        return cls(
            request_id=output.request_id,
            outputs=[InternalCompletionDelta.from_engine(delta) for delta in output.outputs],
            prompt_token_ids=list(output.prompt_token_ids),
            finished=output.finished,
        )

    def to_engine(self) -> GenerateOutput:
        return GenerateOutput(
            request_id=self.request_id,
            outputs=tuple(delta.to_engine() for delta in self.outputs),
            prompt_token_ids=tuple(self.prompt_token_ids),
            finished=self.finished,
        )


class EncodeRequest(BaseModel):
    prompt_token_ids: list[int]
    params: InternalEncodeParams
    request_id: str


class EncodeResponse(BaseModel):
    data: list[float]
    prompt_token_ids: list[int]

    @classmethod
    def from_engine(cls, output: EmbeddingOutput) -> EncodeResponse:
        return cls(data=output.data, prompt_token_ids=list(output.prompt_token_ids))

    def to_engine(self) -> EmbeddingOutput:
        return EmbeddingOutput(data=self.data, prompt_token_ids=tuple(self.prompt_token_ids))


class TokenizeChatRequest(BaseModel):
    messages: list[dict] = Field(default_factory=list)
    tools: list[dict] | None = None
    reasoning_effort: ReasoningEffort | None = None


class TokenizeTextRequest(BaseModel):
    text: str


class PrepareInfillRequest(BaseModel):
    prefix: str
    suffix: str


class TokenIdsResponse(BaseModel):
    token_ids: list[int]


class PreparedInfillPromptResponse(BaseModel):
    prompt_token_ids: list[int]
    strategy: str
    internal_stop: list[str] = Field(default_factory=list)
    response_open_tag: str | None = None
    response_close_tag: str | None = None
    supports_streaming: bool = True
    supports_logprobs: bool = True

    @classmethod
    def from_engine(cls, prompt: PreparedInfillPrompt) -> PreparedInfillPromptResponse:
        return cls(
            prompt_token_ids=list(prompt.prompt_token_ids),
            strategy=prompt.strategy,
            internal_stop=list(prompt.internal_stop),
            response_open_tag=prompt.response_open_tag,
            response_close_tag=prompt.response_close_tag,
            supports_streaming=prompt.supports_streaming,
            supports_logprobs=prompt.supports_logprobs,
        )

    def to_engine(self) -> PreparedInfillPrompt:
        return PreparedInfillPrompt(
            prompt_token_ids=tuple(self.prompt_token_ids),
            strategy=self.strategy,  # type: ignore[arg-type]
            internal_stop=tuple(self.internal_stop),
            response_open_tag=self.response_open_tag,
            response_close_tag=self.response_close_tag,
            supports_streaming=self.supports_streaming,
            supports_logprobs=self.supports_logprobs,
        )


class DecodeTokensRequest(BaseModel):
    token_ids: list[int]


class DecodeTokensResponse(BaseModel):
    text: str


class LoRAAdapterPayload(BaseModel):
    name: str
    path: str
    id: int

    @classmethod
    def from_engine(cls, adapter: LoRAAdapterInfo) -> LoRAAdapterPayload:
        return cls(name=adapter.name, path=adapter.path, id=adapter.id)

    def to_engine(self) -> LoRAAdapterInfo:
        return LoRAAdapterInfo(name=self.name, path=self.path, id=self.id)


class LoRAListResponse(BaseModel):
    data: list[LoRAAdapterPayload]


class LoadLoRARequest(BaseModel):
    name: str
    path: str


class UnloadLoRARequest(BaseModel):
    name: str


class EngineHealthResponse(BaseModel):
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
    loaded_loras: list[str] = Field(default_factory=list)
    transport: str = "unknown"
    available_tool_parsers: list[str] = Field(default_factory=list)
    configured_tool_parser: str | None = None
    effective_tool_parser: str | None = None
    tool_parser_resolution_error: str | None = None
    available_reasoning_parsers: list[str] = Field(default_factory=list)
    configured_reasoning_parser: str | None = None
    effective_reasoning_parser: str | None = None
    reasoning_parser_resolution_error: str | None = None
    configured_reasoning_effort_map: (
        dict[str, dict[str, bool | str | int | float | None] | None] | None
    ) = None
    configured_fim_policy: str | None = None
    effective_fim_mode: EffectiveFIMMode | None = None
    available_infill_strategies: list[InfillStrategy] = Field(default_factory=list)
    configured_native_infill_profile: str | None = None
    configured_fallback_infill_strategy: str | None = None

    @classmethod
    def from_engine(cls, health: EngineHealth) -> EngineHealthResponse:
        return cls(
            ready=health.ready,
            runtime_model=health.runtime_model,
            max_model_len=health.max_model_len,
            weights=health.weights,
            tokenizer_id=health.tokenizer_id,
            quantization=health.quantization,
            dtype=health.dtype,
            tensor_parallel_size=health.tensor_parallel_size,
            pipeline_parallel_size=health.pipeline_parallel_size,
            engine_version=health.engine_version,
            loaded_loras=list(health.loaded_loras),
            transport=health.transport,
            available_tool_parsers=list(health.available_tool_parsers),
            configured_tool_parser=health.configured_tool_parser,
            effective_tool_parser=health.effective_tool_parser,
            tool_parser_resolution_error=health.tool_parser_resolution_error,
            available_reasoning_parsers=list(health.available_reasoning_parsers),
            configured_reasoning_parser=health.configured_reasoning_parser,
            effective_reasoning_parser=health.effective_reasoning_parser,
            reasoning_parser_resolution_error=health.reasoning_parser_resolution_error,
            configured_reasoning_effort_map=health.configured_reasoning_effort_map,
            configured_fim_policy=health.configured_fim_policy,
            effective_fim_mode=health.effective_fim_mode,
            available_infill_strategies=list(health.available_infill_strategies),
            configured_native_infill_profile=health.configured_native_infill_profile,
            configured_fallback_infill_strategy=health.configured_fallback_infill_strategy,
        )

    def to_engine(self) -> EngineHealth:
        return EngineHealth(
            ready=self.ready,
            runtime_model=self.runtime_model,
            max_model_len=self.max_model_len,
            weights=self.weights,
            tokenizer_id=self.tokenizer_id,
            quantization=self.quantization,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            engine_version=self.engine_version,
            loaded_loras=tuple(self.loaded_loras),
            transport=self.transport,
            available_tool_parsers=tuple(self.available_tool_parsers),
            configured_tool_parser=self.configured_tool_parser,
            effective_tool_parser=self.effective_tool_parser,
            tool_parser_resolution_error=self.tool_parser_resolution_error,
            available_reasoning_parsers=tuple(self.available_reasoning_parsers),
            configured_reasoning_parser=self.configured_reasoning_parser,
            effective_reasoning_parser=self.effective_reasoning_parser,
            reasoning_parser_resolution_error=self.reasoning_parser_resolution_error,
            configured_reasoning_effort_map=self.configured_reasoning_effort_map,
            configured_fim_policy=self.configured_fim_policy,
            effective_fim_mode=self.effective_fim_mode,
            available_infill_strategies=tuple(self.available_infill_strategies),
            configured_native_infill_profile=self.configured_native_infill_profile,
            configured_fallback_infill_strategy=self.configured_fallback_infill_strategy,
        )
