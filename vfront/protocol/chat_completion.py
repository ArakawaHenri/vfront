"""Chat Completions protocol — strictly matches OpenAI spec."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from vfront.protocol.common import (
    ChatMessage,
    ChatUsageInfo,
    LogprobContent,
    ResponseFormat,
    StreamOptions,
)

# ── Request ──


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class ImageUrlDetail(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImageUrlContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlDetail


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


MessageContentPart = TextContent | ImageUrlContent | dict[str, Any]


class ChatCompletionRequestMessage(BaseModel):
    role: str
    content: str | list[MessageContentPart] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class DeprecatedFunctionDefinition(BaseModel):
    """Deprecated: use ToolDefinition instead."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionRequestMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None
    seed: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: ResponseFormat | None = None
    service_tier: str | None = None
    metadata: dict[str, str] | None = None
    store: bool | None = None
    reasoning_effort: str | None = None

    # ── Deprecated fields (mapped to tools/tool_choice internally) ──
    function_call: str | dict[str, Any] | None = None
    functions: list[DeprecatedFunctionDefinition] | None = None

    # ── Cloud-only fields (accepted but ignored for SDK compatibility) ──
    audio: dict[str, Any] | None = None
    modalities: list[str] | None = None
    prediction: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    safety_identifier: str | None = None
    verbosity: str | None = None
    web_search_options: dict[str, Any] | None = None


# ── Response ──


class ChoiceLogprobs(BaseModel):
    content: list[LogprobContent] | None = None
    refusal: list[LogprobContent] | None = None


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatUsageInfo | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class StoredChatCompletionObject(ChatCompletionResponse):
    """Chat completion stored object — echoes back relevant request-context fields.

    Returned by GET /chat/completions and GET /chat/completions/{id}.
    All request-context fields default to None so old stored objects
    (serialised before this schema existed) can be deserialised cleanly.
    """

    metadata: dict[str, str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    input_user: str | None = None
    reasoning_effort: str | None = None


class UpdateStoredChatCompletionRequest(BaseModel):
    metadata: dict[str, str]


class ChatCompletionDeleted(BaseModel):
    object: Literal["chat.completion.deleted"] = "chat.completion.deleted"
    id: str
    deleted: bool = True


class ChatCompletionListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[StoredChatCompletionObject] = Field(default_factory=list)
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


class ChatCompletionMessageListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[dict] = Field(default_factory=list)
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


# ── Streaming ──


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    refusal: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: ChatUsageInfo | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
