"""Shared helpers for engine chat tokenization calls."""

from __future__ import annotations

from typing import Any

from vfront.shared.engine.reasoning import ReasoningEffort
from vfront.shared.engine.signatures import callable_accepts_keyword_argument


async def tokenize_chat_messages(
    runtime: Any,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> list[int]:
    """Call ``runtime.tokenize_chat`` while remaining compatible with older fakes."""
    tokenize_chat = runtime.tokenize_chat
    kwargs: dict[str, object] = {}
    if tools is not None and callable_accepts_keyword_argument(tokenize_chat, "tools"):
        kwargs["tools"] = tools
    if callable_accepts_keyword_argument(tokenize_chat, "reasoning_effort"):
        kwargs["reasoning_effort"] = reasoning_effort

    if kwargs:
        return await tokenize_chat(messages, **kwargs)
    return await tokenize_chat(messages)
