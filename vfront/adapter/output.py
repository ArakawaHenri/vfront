"""Translate engine outputs to OpenAI response objects."""

from __future__ import annotations

import base64
import struct
import time
import uuid

from vfront.adapter.assistant_output import normalize_assistant_output
from vfront.adapter.tool_calls import build_tool_calls_from_parsed
from vfront.adapter.usage import estimate_reasoning_tokens
from vfront.backend_identity.fingerprint import get_fingerprint
from vfront.protocol.chat_completion import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChoiceLogprobs,
)
from vfront.protocol.common import (
    ChatMessage,
    ChatUsageInfo,
    CompletionTokensDetails,
    LogprobContent,
    PromptTokensDetails,
    TopLogprob,
    UsageInfo,
)
from vfront.protocol.completion import (
    CompletionChoice,
    CompletionLogprobs,
    CompletionResponse,
)
from vfront.protocol.embedding import EmbeddingData, EmbeddingResponse
from vfront.shared.engine.types import CompletionDelta, EmbeddingOutput, GenerateOutput


def generate_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def generate_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def generate_system_fingerprint(*, lora_name: str | None = None) -> str | None:
    """Return a stable fingerprint for the resolved backend identity."""
    return get_fingerprint(lora_name=lora_name)


def _resolve_system_fingerprint(
    system_fingerprint: str | None,
    *,
    lora_name: str | None,
) -> str | None:
    if system_fingerprint is not None:
        return system_fingerprint
    return generate_system_fingerprint(lora_name=lora_name)


def _convert_logprobs(output: CompletionDelta) -> ChoiceLogprobs | None:
    if output.logprobs is None:
        return None
    content: list[LogprobContent] = []
    for lp in output.logprobs:
        top_list: list[TopLogprob] = []
        if lp:
            for token_str, logprob_val in lp.items():
                top_list.append(
                    TopLogprob(
                        token=token_str,
                        logprob=logprob_val,
                        bytes=list(token_str.encode("utf-8")),
                    )
                )
            if top_list:
                content.append(
                    LogprobContent(
                        token=top_list[0].token,
                        logprob=top_list[0].logprob,
                        bytes=top_list[0].bytes,
                        top_logprobs=top_list,
                    )
                )
    return ChoiceLogprobs(content=content) if content else None

def request_output_to_chat_response(
    output: GenerateOutput,
    model: str,
    request_id: str | None = None,
    has_tool_definitions: bool = False,
    semantic_tool_parsing: bool = False,
    lora_name: str | None = None,
    system_fingerprint: str | None = None,
) -> ChatCompletionResponse:
    choices: list[ChatCompletionChoice] = []
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    for completion in output.outputs:
        total_completion_tokens += len(completion.token_ids)
        total_reasoning_tokens += estimate_reasoning_tokens(completion.reasoning_text)
        finish_reason = _map_finish_reason(completion.finish_reason)
        tool_calls = None
        content: str | None = completion.text
        if completion.parsed_tool_calls is not None:
            # Semantic path: prioritized engine-side results
            tool_calls = build_tool_calls_from_parsed(completion.parsed_tool_calls)
            content = (
                completion.semantic.content_delta if completion.semantic else None
            ) or completion.text
            finish_reason = "tool_calls"
        elif semantic_tool_parsing:
            # Enforced semantic mode: if we reach here, the engine failed to provide
            # parsed_tool_calls even though we expected them (or it's just plain text).
            normalized = normalize_assistant_output(
                completion.text,
                has_tool_definitions=has_tool_definitions,
            )
            if normalized.kind == "tool_calls":
                raise RuntimeError(
                    "Engine-side semantic parser was expected to provide parsed tool calls, "
                    "but only legacy post-hoc tool-call text was available. Check model config."
                )
            content = completion.text
        else:
            # Legacy path: post-hoc regex normalization
            normalized = normalize_assistant_output(
                completion.text,
                has_tool_definitions=has_tool_definitions,
            )
            tool_calls = normalized.tool_calls
            content = normalized.text
            if normalized.kind == "tool_calls":
                finish_reason = "tool_calls"
        choices.append(
            ChatCompletionChoice(
                index=completion.index,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    reasoning=completion.reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
                logprobs=_convert_logprobs(completion),
            )
        )
    prompt_tokens = len(output.prompt_token_ids)
    return ChatCompletionResponse(
        id=request_id or generate_request_id(),
        created=int(time.time()),
        model=model,
        choices=choices,
        usage=ChatUsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=prompt_tokens + total_completion_tokens,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=total_reasoning_tokens
            ),
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        ),
        system_fingerprint=_resolve_system_fingerprint(
            system_fingerprint,
            lora_name=lora_name,
        ),
    )


def request_output_to_completion_response(
    output: GenerateOutput,
    model: str,
    request_id: str | None = None,
    echo: bool = False,
    prompt_text: str = "",
    lora_name: str | None = None,
    system_fingerprint: str | None = None,
    completion_tokens_override: int | None = None,
) -> CompletionResponse:
    choices: list[CompletionChoice] = []
    total_completion_tokens = 0
    for completion in output.outputs:
        total_completion_tokens += len(completion.token_ids)
        text = (prompt_text + completion.text) if echo else completion.text
        choices.append(
            CompletionChoice(
                index=completion.index,
                text=text,
                finish_reason=_map_finish_reason(completion.finish_reason),
                logprobs=_convert_completion_logprobs(completion),
            )
        )
    prompt_tokens = len(output.prompt_token_ids)
    return CompletionResponse(
        id=request_id or generate_completion_id(),
        created=int(time.time()),
        model=model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens_override or total_completion_tokens,
            total_tokens=prompt_tokens + (completion_tokens_override or total_completion_tokens),
        ),
        system_fingerprint=_resolve_system_fingerprint(
            system_fingerprint,
            lora_name=lora_name,
        ),
    )


def _convert_completion_logprobs(output: CompletionDelta) -> CompletionLogprobs | None:
    if output.logprobs is None:
        return None
    tokens, token_logprobs = [], []
    top_logprobs_list: list[dict[str, float] | None] = []
    text_offset: list[int] = []
    offset = 0
    for lp in output.logprobs:
        if lp:
            for token_str, logprob_val in lp.items():
                tokens.append(token_str)
                token_logprobs.append(logprob_val)
                top_logprobs_list.append(dict(lp))
                text_offset.append(offset)
                offset += len(token_str)
                break
    return CompletionLogprobs(
        tokens=tokens,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs_list,
        text_offset=text_offset,
    )


def embedding_outputs_to_response(
    outputs: list[EmbeddingOutput],
    model: str,
    encoding_format: str = "float",
) -> EmbeddingResponse:
    data: list[EmbeddingData] = []
    total_prompt_tokens = 0
    for i, output in enumerate(outputs):
        total_prompt_tokens += len(output.prompt_token_ids)
        embedding: list[float] | str = (
            _float_list_to_base64(output.data) if encoding_format == "base64" else output.data
        )
        data.append(EmbeddingData(embedding=embedding, index=i))
    return EmbeddingResponse(
        data=data,
        model=model,
        usage=UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=0,
            total_tokens=total_prompt_tokens,
        ),
    )


def _float_list_to_base64(floats: list[float]) -> str:
    buf = struct.pack(f"<{len(floats)}f", *floats)
    return base64.b64encode(buf).decode("ascii")


def _map_finish_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    return {
        "stop": "stop",
        "length": "length",
        "abort": "content_filter",
        "tool_calls": "tool_calls",
    }.get(reason, reason)
