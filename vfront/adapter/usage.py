"""Shared usage-accounting helpers for adapters."""

from __future__ import annotations


def estimate_reasoning_tokens(reasoning_text: str | None) -> int:
    if not reasoning_text:
        return 0
    return max(1, len(reasoning_text) // 4)
