"""Tokenizer utilities for token counting and prompt encoding."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

from vfront.shared.config.models import ReasoningConfig
from vfront.shared.engine.reasoning import ReasoningEffort
from vfront.shared.engine.reasoning_adapter import (
    resolve_reasoning_template_kwargs,
    supports_chat_template_kwarg,
)

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer import AnyTokenizer


def build_chat_template_kwargs(
    *,
    tokenizer: AnyTokenizer,
    tools: list[dict] | None,
    add_generation_prompt: bool,
    tokenize: bool,
    reasoning_effort: ReasoningEffort | None,
    reasoning_config: ReasoningConfig | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": tokenize,
    }
    if tools and supports_chat_template_kwarg(tokenizer, "tools"):
        kwargs["tools"] = tools
    kwargs.update(
        resolve_reasoning_template_kwargs(
            tokenizer=tokenizer,
            reasoning_effort=reasoning_effort,
            config=reasoning_config,
        )
    )
    return kwargs


def _normalize_messages_for_chat_template(messages: list[dict]) -> list[dict]:
    """Mirror vLLM chat-utils postprocessing for assistant tool-call history.

    Tool-use chat templates increasingly expect assistant-message tool-call
    arguments to be structured objects, not OpenAI-style JSON strings.
    """
    normalized_messages = deepcopy(messages)
    for message in normalized_messages:
        if message.get("role") != "assistant" or "tool_calls" not in message:
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        if len(tool_calls) == 0:
            message.pop("tool_calls", None)
            continue

        for item in tool_calls:
            if not isinstance(item, dict):
                raise ValueError("assistant message 'tool_calls' entries must be objects.")
            function = item.get("function")
            if not isinstance(function, dict):
                raise ValueError(
                    "assistant message 'tool_calls[].function' must be an object."
                )
            arguments = function.get("arguments")
            if arguments:
                if not isinstance(arguments, (dict, list)):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "assistant message tool call 'function.arguments' "
                            "must be valid JSON."
                        ) from exc
            else:
                function["arguments"] = {}

    return normalized_messages


def count_tokens(tokenizer: AnyTokenizer, text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))


def encode_prompt(tokenizer: AnyTokenizer, text: str) -> list[int]:
    """Encode a text string to token IDs."""
    return tokenizer.encode(text)


def apply_chat_template(
    tokenizer: AnyTokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    reasoning_effort: ReasoningEffort | None = None,
    reasoning_config: ReasoningConfig | None = None,
) -> str:
    """Apply the tokenizer's chat template to messages."""
    kwargs = build_chat_template_kwargs(
        tokenizer=tokenizer,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        reasoning_config=reasoning_config,
    )
    return tokenizer.apply_chat_template(_normalize_messages_for_chat_template(messages), **kwargs)


def apply_chat_template_to_ids(
    tokenizer: AnyTokenizer,
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    reasoning_effort: ReasoningEffort | None = None,
    reasoning_config: ReasoningConfig | None = None,
) -> list[int]:
    """Apply the tokenizer's chat template and return token IDs directly."""
    kwargs = build_chat_template_kwargs(
        tokenizer=tokenizer,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        reasoning_effort=reasoning_effort,
        reasoning_config=reasoning_config,
    )
    return tokenizer.apply_chat_template(_normalize_messages_for_chat_template(messages), **kwargs)
