"""Responses API protocol models — strictly matches OpenAI Python SDK types/responses/*.

Reference: https://github.com/openai/openai-python/tree/main/src/openai/types/responses
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Shared enums / literals
# ---------------------------------------------------------------------------

ResponseStatus = Literal[
    "completed",
    "failed",
    "in_progress",
    "incomplete",
    "cancelling",
    "cancelled",
    "queued",
]


# ---------------------------------------------------------------------------
# Text configuration (structured outputs)
# ---------------------------------------------------------------------------


class ResponseTextConfigJsonSchema(BaseModel):
    name: str
    description: str | None = None
    schema_: dict | None = Field(None, alias="schema")
    strict: bool | None = None
    model_config = {"populate_by_name": True}


class ResponseTextFormat(BaseModel):
    """The nested 'format' object inside 'text' — matches official SDK."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: ResponseTextConfigJsonSchema | None = None


class ResponseTextConfig(BaseModel):
    """text config in the request/response — wraps format in a 'format' key.

    Official shape: {"text": {"format": {"type": "json_schema", ...}}}

    Backward-compat: the old flat shape {"type": ..., "json_schema": ...} is
    also accepted via the model_validator and promoted to {"format": {...}}.
    """

    format: ResponseTextFormat = Field(default_factory=ResponseTextFormat)

    @model_validator(mode="before")
    @classmethod
    def _compat_flat_input(cls, data: object) -> object:
        """Promote old flat {type, json_schema} input to {format: {...}}."""
        if isinstance(data, dict) and "type" in data and "format" not in data:
            return {"format": data}
        return data


# ---------------------------------------------------------------------------
# Tools (function calling)
# ---------------------------------------------------------------------------


class FunctionToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict | None = None
    strict: bool | None = None


ToolParam = FunctionToolDefinition  # Only function tools for vLLM

ToolChoiceOptions = Literal["auto", "none", "required"]
ToolChoiceFunction = dict  # {"type": "function", "name": "..."}
ToolChoice = ToolChoiceOptions | ToolChoiceFunction


# ---------------------------------------------------------------------------
# Output content types
# ---------------------------------------------------------------------------


class OutputText(BaseModel):
    """A text output from the model."""

    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: list[Any] = Field(default_factory=list)
    logprobs: list[Any] | None = None


class OutputRefusal(BaseModel):
    """A refusal output from the model."""

    type: Literal["refusal"] = "refusal"
    refusal: str = ""


OutputContent = OutputText | OutputRefusal


# ---------------------------------------------------------------------------
# Output items
# ---------------------------------------------------------------------------


class OutputMessage(BaseModel):
    """An output message from the model."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    content: list[OutputContent] = Field(default_factory=list)


class FunctionToolCall(BaseModel):
    """A function tool call output item."""

    id: str = Field(default_factory=lambda: f"fc_{uuid.uuid4().hex[:24]}")
    type: Literal["function_call"] = "function_call"
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    name: str = ""
    arguments: str = ""
    status: Literal["in_progress", "completed", "incomplete"] | None = "completed"


class ReasoningItem(BaseModel):
    """A reasoning output item (thinking tokens)."""

    id: str = Field(default_factory=lambda: f"rs_{uuid.uuid4().hex[:24]}")
    type: Literal["reasoning"] = "reasoning"
    summary: list[dict] = Field(default_factory=list)


ResponseOutputItem = OutputMessage | FunctionToolCall | ReasoningItem


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class InputTokensDetails(BaseModel):
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens_details: OutputTokensDetails = Field(default_factory=OutputTokensDetails)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ResponseError(BaseModel):
    code: str
    message: str


# ---------------------------------------------------------------------------
# Incomplete details
# ---------------------------------------------------------------------------


class IncompleteDetails(BaseModel):
    reason: Literal["max_output_tokens", "content_filter"] | None = None


# ---------------------------------------------------------------------------
# Response object (the main response model)
# ---------------------------------------------------------------------------


class ResponseObject(BaseModel):
    """The Response object — object: "response"."""

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:24]}")
    object: Literal["response"] = "response"
    created_at: float = Field(default_factory=time.time)
    status: ResponseStatus = "in_progress"
    model: str = ""

    output: list[ResponseOutputItem] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    error: ResponseError | None = None
    incomplete_details: IncompleteDetails | None = None
    metadata: dict[str, str] | None = None

    # Echo-back params
    instructions: str | list | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: str | list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    text: ResponseTextConfig | None = None
    tools: list[FunctionToolDefinition] | None = None
    tool_choice: ToolChoice | None = "auto"
    parallel_tool_calls: bool = True
    store: bool | None = None
    previous_response_id: str | None = None
    truncation: Literal["auto", "disabled"] | None = None
    user: str | None = None
    background: bool | None = None
    top_logprobs: int | None = None
    reasoning: dict | None = None
    service_tier: str | None = None
    completed_at: float | None = None
    max_tool_calls: int | None = None


# ---------------------------------------------------------------------------
# Deleted response
# ---------------------------------------------------------------------------


class ResponseDeleted(BaseModel):
    id: str
    object: Literal["response"] = "response"
    deleted: bool = True


class ResponseInputMessageResource(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["user", "system", "developer"] = "user"
    status: Literal["in_progress", "completed", "incomplete"] | None = "completed"
    content: list[InputContent] = Field(default_factory=list)


class ResponseFunctionToolCallOutputResource(BaseModel):
    id: str
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


ResponseItemResource = (
    ResponseInputMessageResource
    | OutputMessage
    | FunctionToolCall
    | ResponseFunctionToolCallOutputResource
)


class ResponseItemList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ResponseItemResource] = Field(default_factory=list)
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ResponseInputTextContent(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class ResponseInputImageContent(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: Literal["auto", "low", "high"] | None = None


class ResponseOutputTextContent(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


InputContent = (
    ResponseInputTextContent | ResponseInputImageContent | ResponseOutputTextContent | str
)


class ResponseInputItem(BaseModel):
    """A single input item (turn) in the conversation."""

    role: Literal["user", "assistant", "system", "developer"] = "user"
    content: str | list[InputContent] = ""
    type: Literal["message"] | None = None
    model_config = {"extra": "allow"}


class FunctionCallOutput(BaseModel):
    """A function call output item in the input (for multi-turn)."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


ResponseInput = str | list[ResponseInputItem | FunctionCallOutput | dict]


class ResponseStreamOptions(BaseModel):
    """Stream options for Responses API."""

    include_obfuscation: bool = False


class ResponseCreateRequest(BaseModel):
    """POST /v1/responses request body."""

    model: str
    input: ResponseInput = ""

    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    # ── Extension fields (not in OpenAI SDK, but useful for local LLM) ──
    stop: str | list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    stream: bool | None = False
    stream_options: ResponseStreamOptions | None = None
    store: bool | None = None
    metadata: dict[str, str] | None = None

    tools: list[FunctionToolDefinition] | None = None
    tool_choice: ToolChoice | None = "auto"
    parallel_tool_calls: bool | None = True

    text: ResponseTextConfig | None = None
    truncation: Literal["auto", "disabled"] | None = None
    previous_response_id: str | None = None
    background: bool | None = None
    user: str | None = None
    top_logprobs: int | None = None

    # Reasoning (for reasoning models like DeepSeek-R1)
    reasoning: dict | None = None

    # Service tier (echo-back only)
    service_tier: str | None = None

    # Limit tool call rounds
    max_tool_calls: int | None = None

    # ── Cloud-only fields (accepted but ignored for SDK compatibility) ──
    include: list[str] | None = None
    prompt: dict | None = None
    context_management: list[dict] | None = None
    conversation: str | dict | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    safety_identifier: str | None = None


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


class ResponseCreatedEvent(BaseModel):
    type: Literal["response.created"] = "response.created"
    response: ResponseObject
    sequence_number: int = 0


class ResponseInProgressEvent(BaseModel):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: ResponseObject
    sequence_number: int = 0


class ResponseCompletedEvent(BaseModel):
    type: Literal["response.completed"] = "response.completed"
    response: ResponseObject
    sequence_number: int = 0


class ResponseFailedEvent(BaseModel):
    type: Literal["response.failed"] = "response.failed"
    response: ResponseObject
    sequence_number: int = 0


class ResponseIncompleteEvent(BaseModel):
    type: Literal["response.incomplete"] = "response.incomplete"
    response: ResponseObject
    sequence_number: int = 0


class OutputItemAddedEvent(BaseModel):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    item: ResponseOutputItem
    output_index: int = 0
    sequence_number: int = 0


class OutputItemDoneEvent(BaseModel):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    item: ResponseOutputItem
    output_index: int = 0
    sequence_number: int = 0


class ContentPartAddedEvent(BaseModel):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    part: OutputContent
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    sequence_number: int = 0


class ContentPartDoneEvent(BaseModel):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    part: OutputContent
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    sequence_number: int = 0


class TextDeltaEvent(BaseModel):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    delta: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    logprobs: list[Any] = Field(default_factory=list)
    sequence_number: int = 0


class TextDoneEvent(BaseModel):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    text: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    logprobs: list[Any] | None = None
    sequence_number: int = 0


class FunctionCallArgumentsDeltaEvent(BaseModel):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    delta: str = ""
    item_id: str = ""
    output_index: int = 0
    sequence_number: int = 0


class FunctionCallArgumentsDoneEvent(BaseModel):
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    name: str = ""
    arguments: str = ""
    item_id: str = ""
    output_index: int = 0
    sequence_number: int = 0


class RefusalDeltaEvent(BaseModel):
    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    delta: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    sequence_number: int = 0


class RefusalDoneEvent(BaseModel):
    type: Literal["response.refusal.done"] = "response.refusal.done"
    refusal: str = ""
    item_id: str = ""
    output_index: int = 0
    content_index: int = 0
    sequence_number: int = 0


class ResponseErrorEvent(BaseModel):
    type: Literal["response.error"] = "response.error"
    code: str = ""
    message: str = ""
    sequence_number: int = 0


ResponseStreamEvent = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | OutputItemAddedEvent
    | OutputItemDoneEvent
    | ContentPartAddedEvent
    | ContentPartDoneEvent
    | TextDeltaEvent
    | TextDoneEvent
    | FunctionCallArgumentsDeltaEvent
    | FunctionCallArgumentsDoneEvent
    | RefusalDeltaEvent
    | RefusalDoneEvent
    | ResponseErrorEvent
)
