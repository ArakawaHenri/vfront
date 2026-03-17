"""Shared helpers for reasoning-effort compatibility."""

from __future__ import annotations

from typing import Literal, TypeGuard

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
ReasoningEffortKey = ReasoningEffort | None

VALID_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)
_VALID_REASONING_EFFORTS = frozenset(VALID_REASONING_EFFORTS)


def is_reasoning_effort(value: object) -> TypeGuard[ReasoningEffort]:
    """Return True when ``value`` is a supported OpenAI reasoning effort."""
    return isinstance(value, str) and value in _VALID_REASONING_EFFORTS


def expect_reasoning_effort(value: object) -> ReasoningEffort:
    """Return ``value`` as a typed reasoning effort or raise ``ValueError``."""
    if not is_reasoning_effort(value):
        raise ValueError(
            "reasoning_effort must be one of "
            "'none', 'minimal', 'low', 'medium', 'high', or 'xhigh'."
        )
    return value


def is_reasoning_effort_key(value: object) -> TypeGuard[ReasoningEffortKey]:
    """Return True when ``value`` is a supported effort or the implicit default."""
    return value is None or is_reasoning_effort(value)


def format_reasoning_effort_key(value: ReasoningEffortKey) -> str:
    """Return a stable display label for a reasoning-effort config key."""
    return "~" if value is None else value


def extract_reasoning_effort(settings: object) -> ReasoningEffort | None:
    """Extract ``reasoning.effort`` from a Responses-style config dict."""
    if not isinstance(settings, dict):
        return None
    effort = settings.get("effort")
    if is_reasoning_effort(effort):
        return effort
    return None
