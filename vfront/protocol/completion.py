"""Legacy Completions protocol — strictly matches OpenAI spec."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, StrictInt

from vfront.protocol.common import StreamOptions, UsageInfo

FinishReason = Literal["stop", "length", "content_filter"]


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str] | list[StrictInt] | list[list[StrictInt]] | None
    suffix: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    logprobs: int | None = None
    echo: bool | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    best_of: int | None = None
    logit_bias: dict[str, int] | None = None
    user: str | None = None
    seed: int | None = None


class CompletionLogprobs(BaseModel):
    text_offset: list[int] | None = None
    token_logprobs: list[float | None] | None = None
    tokens: list[str] | None = None
    top_logprobs: list[dict[str, float] | None] | None = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: FinishReason | None = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None


class CompletionStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: FinishReason | None = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChoice]
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None
