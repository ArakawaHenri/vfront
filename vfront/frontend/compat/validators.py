"""Unified request validators for Chat, Responses, and Completions endpoints.

Each validator raises ``InvalidRequestError`` for unsupported or invalid fields.
Route handlers call the appropriate validator near the top of the handler,
before any engine interaction.
"""

from __future__ import annotations

from vfront.frontend.api.v1.helper import validate_completion_best_of
from vfront.frontend.compat.capabilities import DEFAULT_CAPABILITIES, ServerCapabilities
from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.protocol.chat_completion import ChatCompletionRequest
from vfront.protocol.completion import CompletionRequest
from vfront.protocol.response import ResponseCreateRequest
from vfront.shared.engine.reasoning import is_reasoning_effort


def validate_chat_request(
    request: ChatCompletionRequest,
    caps: ServerCapabilities = DEFAULT_CAPABILITIES,
) -> None:
    """Raise InvalidRequestError for any invalid or unsupported chat parameter."""
    del caps
    if not request.messages:
        raise InvalidRequestError("'messages' must be a non-empty array.")

    if request.temperature is not None and not (0.0 <= request.temperature <= 2.0):
        raise InvalidRequestError(
            f"'temperature' must be between 0 and 2, got {request.temperature}.",
            param="temperature",
        )
    if request.top_p is not None and not (0.0 <= request.top_p <= 1.0):
        raise InvalidRequestError(
            f"'top_p' must be between 0 and 1, got {request.top_p}.",
            param="top_p",
        )
    if request.n is not None and not (1 <= request.n <= 128):
        raise InvalidRequestError(
            f"'n' must be between 1 and 128, got {request.n}.",
            param="n",
        )
    if request.top_logprobs is not None and not (0 <= request.top_logprobs <= 20):
        raise InvalidRequestError(
            f"'top_logprobs' must be between 0 and 20, got {request.top_logprobs}.",
            param="top_logprobs",
        )
    if request.top_logprobs is not None and request.logprobs is not True:
        raise InvalidRequestError(
            "'logprobs' must be set to true if 'top_logprobs' is used.",
            param="top_logprobs",
        )

    # Cloud-only compatibility fields are accepted and ignored so OpenAI SDK
    # requests continue to work against local deployments without route-level 400s.

    if request.tool_choice is not None:
        if isinstance(request.tool_choice, str):
            if request.tool_choice not in ("none", "auto", "required"):
                raise InvalidRequestError(
                    f"'tool_choice' string must be 'none', 'auto', or 'required', "
                    f"got '{request.tool_choice}'.",
                    param="tool_choice",
                )
        elif isinstance(request.tool_choice, dict):
            if request.tool_choice.get("type") != "function":
                raise InvalidRequestError(
                    "'tool_choice' object must have type 'function'.",
                    param="tool_choice",
                )
            fn = request.tool_choice.get("function")
            if not isinstance(fn, dict) or not fn.get("name"):
                raise InvalidRequestError(
                    "'tool_choice' object must specify 'function.name'.",
                    param="tool_choice",
                )

    if request.reasoning_effort is not None and not is_reasoning_effort(
        request.reasoning_effort
    ):
        raise InvalidRequestError(
            "'reasoning_effort' must be one of 'none', 'minimal', 'low', "
            "'medium', 'high', or 'xhigh'.",
            param="reasoning_effort",
        )


def validate_responses_request(
    request: ResponseCreateRequest,
    caps: ServerCapabilities = DEFAULT_CAPABILITIES,
) -> None:
    """Raise InvalidRequestError for any invalid or unsupported Responses parameter."""
    del caps
    if request.temperature is not None and not (0.0 <= request.temperature <= 2.0):
        raise InvalidRequestError(
            f"'temperature' must be between 0 and 2, got {request.temperature}.",
            param="temperature",
        )
    if request.top_p is not None and not (0.0 <= request.top_p <= 1.0):
        raise InvalidRequestError(
            f"'top_p' must be between 0 and 1, got {request.top_p}.",
            param="top_p",
        )
    if request.max_output_tokens is not None and request.max_output_tokens < 1:
        raise InvalidRequestError(
            f"'max_output_tokens' must be greater than or equal to 1, got {request.max_output_tokens}.",
            param="max_output_tokens",
        )
    if request.top_logprobs is not None and not (0 <= request.top_logprobs <= 20):
        raise InvalidRequestError(
            f"'top_logprobs' must be between 0 and 20, got {request.top_logprobs}.",
            param="top_logprobs",
        )
    if request.max_tool_calls is not None and request.max_tool_calls < 1:
        raise InvalidRequestError(
            f"'max_tool_calls' must be greater than or equal to 1, got {request.max_tool_calls}.",
            param="max_tool_calls",
        )

    # Cloud-only compatibility fields are accepted and ignored so OpenAI SDK
    # requests continue to work against local deployments without route-level 400s.

    if (
        request.reasoning is not None
        and isinstance(request.reasoning, dict)
        and (effort := request.reasoning.get("effort")) is not None
        and not is_reasoning_effort(effort)
    ):
        raise InvalidRequestError(
            "'reasoning.effort' must be one of 'none', 'minimal', 'low', "
            "'medium', 'high', or 'xhigh'.",
            param="reasoning.effort",
        )


def validate_completion_request(
    request: CompletionRequest,
    *,
    num_prompts: int = 1,
    caps: ServerCapabilities = DEFAULT_CAPABILITIES,
) -> None:
    """Raise InvalidRequestError for any invalid or unsupported Completions parameter.

    Args:
        request:     The parsed completion request.
        num_prompts: Number of prompts after normalization (used for multi-prompt checks).
        caps:        Server capabilities (defaults to DEFAULT_CAPABILITIES).
    """
    validate_completion_prompt_shape(request.prompt)

    if request.temperature is not None and not (0.0 <= request.temperature <= 2.0):
        raise InvalidRequestError(
            f"'temperature' must be between 0 and 2, got {request.temperature}.",
            param="temperature",
        )
    if request.top_p is not None and not (0.0 <= request.top_p <= 1.0):
        raise InvalidRequestError(
            f"'top_p' must be between 0 and 1, got {request.top_p}.",
            param="top_p",
        )
    if request.n is not None and not (1 <= request.n <= 128):
        raise InvalidRequestError(
            f"'n' must be between 1 and 128, got {request.n}.",
            param="n",
        )
    if request.max_tokens is not None and request.max_tokens < 0:
        raise InvalidRequestError(
            f"'max_tokens' must be greater than or equal to 0, got {request.max_tokens}.",
            param="max_tokens",
        )
    if request.logprobs is not None and not (0 <= request.logprobs <= 5):
        raise InvalidRequestError(
            f"'logprobs' must be between 0 and 5, got {request.logprobs}.",
            param="logprobs",
        )
    if request.presence_penalty is not None and not (-2.0 <= request.presence_penalty <= 2.0):
        raise InvalidRequestError(
            "'presence_penalty' must be between -2 and 2, "
            f"got {request.presence_penalty}.",
            param="presence_penalty",
        )
    if request.frequency_penalty is not None and not (-2.0 <= request.frequency_penalty <= 2.0):
        raise InvalidRequestError(
            "'frequency_penalty' must be between -2 and 2, "
            f"got {request.frequency_penalty}.",
            param="frequency_penalty",
        )
    validate_completion_best_of(request)

    if not caps.supports_completion_multi_prompt and num_prompts > 1:
        raise InvalidRequestError(
            "Multiple prompts are not supported for this /v1/completions implementation.",
            param="prompt",
        )

    if (
        not caps.supports_completion_stream_multi_prompt
        and request.stream
        and num_prompts > 1
    ):
        raise InvalidRequestError(
            "Streaming is not supported for multi-prompt requests. "
            "Set 'stream' to false or pass a single prompt.",
            param="stream",
        )


def validate_completion_prompt_shape(
    prompt: str | list[str] | list[int] | list[list[int]] | None,
) -> None:
    if prompt is None or isinstance(prompt, str):
        return
    if not isinstance(prompt, list) or not prompt:
        raise InvalidRequestError(
            "'prompt' must be a non-empty string, array of strings, "
            "array of integers, or array of token arrays.",
            param="prompt",
        )
    if all(type(item) is int for item in prompt):
        return
    if all(isinstance(item, str) for item in prompt):
        return
    if all(
        isinstance(item, list) and all(type(token) is int for token in item)
        for item in prompt
    ):
        return
    raise InvalidRequestError(
        "'prompt' must be a string, array of strings, array of integers, "
        "or array of token arrays.",
        param="prompt",
    )
