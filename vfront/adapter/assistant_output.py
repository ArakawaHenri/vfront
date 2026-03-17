"""Unified normalization for assistant text and tool-call outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vfront.adapter.tool_calls import build_tool_calls_from_text
from vfront.protocol.common import ToolCall

AssistantOutputKind = Literal["text", "tool_calls"]
AssistantStreamKind = Literal["undecided", "text", "tool_calls"]


@dataclass(slots=True, frozen=True)
class AssistantOutput:
    """Normalized assistant output independent of API surface."""

    kind: AssistantOutputKind
    text: str | None = None
    tool_calls: list[ToolCall] | None = None


def looks_like_tool_call_prefix(text: str) -> bool:
    """Return True when the text could still become a tool-call payload."""
    stripped = text.lstrip()
    return stripped.startswith("[") or stripped.startswith("<tool_call>")


def normalize_assistant_output(
    text: str,
    has_tool_definitions: bool = False,
) -> AssistantOutput:
    """Normalize model text into either plain text or tool calls."""
    if has_tool_definitions:
        tool_calls = parse_tool_calls_if_present(text)
        if tool_calls:
            return AssistantOutput(kind="tool_calls", tool_calls=tool_calls)
    return AssistantOutput(kind="text", text=text)


def parse_tool_calls_if_present(text: str) -> list[ToolCall] | None:
    """Parse tool calls only when the text looks like a tool payload."""
    if not looks_like_tool_call_prefix(text):
        return None
    return build_tool_calls_from_text(text)


@dataclass(slots=True)
class AssistantOutputStreamState:
    """Track streaming state until an assistant output can be normalized."""

    has_tool_definitions: bool = False
    kind: AssistantStreamKind = "undecided"
    text: str = ""
    streamed_text: str = ""
    finish_reason: str | None = None

    def update(self, new_text: str, finish_reason: str | None = None) -> str | None:
        """Accept the latest cumulative text and return any safe text delta to emit."""
        previous_text = self.text
        delta_text = new_text[len(previous_text) :]
        self.text = new_text

        if self.kind == "undecided" and new_text.strip():
            if self.has_tool_definitions and looks_like_tool_call_prefix(new_text):
                self.kind = "tool_calls"
            else:
                self.kind = "text"

        if finish_reason is not None:
            self.finish_reason = finish_reason

        if delta_text and self.kind == "text":
            self.streamed_text += delta_text
            return delta_text
        return None

    def finalize(self) -> tuple[AssistantOutput, str | None]:
        """Return normalized output plus any deferred plain-text chunk."""
        normalized = normalize_assistant_output(
            self.text,
            has_tool_definitions=self.has_tool_definitions,
        )
        deferred_text = None
        if normalized.kind == "text" and normalized.text and not self.streamed_text:
            deferred_text = normalized.text
        return normalized, deferred_text
