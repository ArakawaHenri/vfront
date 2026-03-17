"""Translate OpenAI request parameters to engine-agnostic GenerateParams."""

from __future__ import annotations

from typing import Any

from vfront.protocol.chat_completion import ChatCompletionRequest
from vfront.protocol.completion import CompletionRequest
from vfront.shared.engine.reasoning import expect_reasoning_effort
from vfront.shared.engine.types import GenerateParams


def extract_multimodal_data(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract multimodal image data from serialized chat message dicts.

    Scans content parts for ``{"type": "image_url", ...}`` entries and returns
    ``{"image": [url, ...]}`` suitable for passing to vLLM's ``multi_modal_data``.
    Returns ``None`` when no image parts are found.
    """
    images: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
            if url:
                images.append(url)
    return {"image": images} if images else None


def chat_request_to_generate_params(
    request: ChatCompletionRequest,
) -> GenerateParams:
    """Convert a ChatCompletionRequest to GenerateParams."""
    params = GenerateParams(chat_template_applied=True)

    params.temperature = request.temperature if request.temperature is not None else 1.0

    if request.top_p is not None:
        params.top_p = request.top_p
    if request.n is not None:
        params.n = request.n

    max_toks = request.max_completion_tokens or request.max_tokens
    if max_toks is not None:
        params.max_tokens = max_toks

    if request.stop is not None:
        params.stop = request.stop if isinstance(request.stop, list) else [request.stop]

    if request.presence_penalty is not None:
        params.presence_penalty = request.presence_penalty
    if request.frequency_penalty is not None:
        params.frequency_penalty = request.frequency_penalty

    if request.logprobs and request.top_logprobs is not None:
        params.logprobs = request.top_logprobs
    elif request.logprobs:
        params.logprobs = 1

    if request.seed is not None:
        params.seed = request.seed

    if request.logit_bias:
        params.logit_bias = {int(k): v for k, v in request.logit_bias.items()}

    if request.response_format is not None:
        if request.response_format.type == "json_object":
            params.json_object = True
        elif (
            request.response_format.type == "json_schema"
            and request.response_format.json_schema is not None
        ):
            params.json_schema = request.response_format.json_schema.schema_ or {}

    if request.reasoning_effort is not None:
        params.reasoning_effort = expect_reasoning_effort(request.reasoning_effort)

    return params


def resolve_deprecated_functions(request: ChatCompletionRequest) -> None:
    """Map deprecated `functions`/`function_call` to `tools`/`tool_choice` in-place.

    Per OpenAI API: `functions` is deprecated in favor of `tools`,
    and `function_call` is deprecated in favor of `tool_choice`.
    """
    if request.functions and not request.tools:
        from vfront.protocol.chat_completion import FunctionDefinition, ToolDefinition

        request.tools = [
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name=fn.name,
                    description=fn.description,
                    parameters=fn.parameters,
                ),
            )
            for fn in request.functions
        ]
    if request.function_call is not None and request.tool_choice is None:
        if isinstance(request.function_call, str):
            request.tool_choice = request.function_call
        elif isinstance(request.function_call, dict) and "name" in request.function_call:
            request.tool_choice = {
                "type": "function",
                "function": {"name": request.function_call["name"]},
            }


def completion_request_to_generate_params(
    request: CompletionRequest,
) -> GenerateParams:
    """Convert a CompletionRequest to GenerateParams."""
    params = GenerateParams()

    params.temperature = request.temperature if request.temperature is not None else 1.0

    if request.top_p is not None:
        params.top_p = request.top_p
    if request.n is not None:
        params.n = request.n

    params.max_tokens = request.max_tokens if request.max_tokens is not None else 16

    if request.stop is not None:
        params.stop = request.stop if isinstance(request.stop, list) else [request.stop]

    if request.presence_penalty is not None:
        params.presence_penalty = request.presence_penalty
    if request.frequency_penalty is not None:
        params.frequency_penalty = request.frequency_penalty

    if request.logprobs is not None and request.logprobs > 0:
        params.logprobs = request.logprobs
    if request.echo:
        params.prompt_logprobs = request.logprobs or 0

    if request.seed is not None:
        params.seed = request.seed
    if request.logit_bias:
        params.logit_bias = {int(k): v for k, v in request.logit_bias.items()}
    if request.best_of is not None and request.best_of > 0:
        params.best_of = request.best_of

    return params
