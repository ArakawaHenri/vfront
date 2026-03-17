"""Common types shared across all endpoints — strictly matches OpenAI spec."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorObject(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorObject


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int = 0
    total_tokens: int


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class ChatUsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobContent(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    refusal: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: str | None = None
    schema_: dict | None = Field(None, alias="schema")
    strict: bool | None = None
    model_config = {"populate_by_name": True}


class ResponseFormat(BaseModel):
    type: str = "text"
    json_schema: JsonSchemaResponseFormat | None = None
