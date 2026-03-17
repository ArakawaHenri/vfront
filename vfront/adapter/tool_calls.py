"""Tool call format conversion utilities."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence

from vfront.protocol.common import FunctionCall, ToolCall
from vfront.shared.engine.types import ParsedToolCall


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def serialize_tool_call_arguments(arguments: object) -> str:
    """Canonicalize tool-call arguments into JSON text when structured."""
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return ""
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def limit_parallel_tool_calls[TToolCall](
    tool_calls: Sequence[TToolCall],
    *,
    parallel_tool_calls: bool | None,
) -> list[TToolCall]:
    """Enforce serial tool-call semantics when parallel tool calls are disabled."""
    if parallel_tool_calls is False:
        return list(tool_calls[:1])
    return list(tool_calls)


def build_tool_calls_from_text(text: str) -> list[ToolCall] | None:
    """Extract tool calls from model output text."""
    calls = _try_json_array_format(text)
    if calls:
        return calls
    calls = _try_xml_format(text)
    if calls:
        return calls
    return None


def build_tool_calls_from_parsed(
    tool_calls: Sequence[ParsedToolCall],
) -> list[ToolCall] | None:
    """Convert engine-side parsed tool calls into protocol tool calls."""
    if not tool_calls:
        return None
    return [
        ToolCall(
            id=item.id or generate_tool_call_id(),
            type=item.type,
            function=FunctionCall(
                name=item.name,
                arguments=serialize_tool_call_arguments(item.arguments),
            ),
        )
        for item in tool_calls
    ]


def _try_json_array_format(text: str) -> list[ToolCall] | None:
    text = text.strip()
    if not text.startswith("["):
        return None
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return None
        calls = []
        for item in data:
            if isinstance(item, dict) and "name" in item:
                args = item.get("arguments", item.get("parameters", {}))
                calls.append(
                    ToolCall(
                        id=generate_tool_call_id(),
                        type="function",
                        function=FunctionCall(
                            name=item["name"],
                            arguments=serialize_tool_call_arguments(args),
                        ),
                    )
                )
        return calls if calls else None
    except (json.JSONDecodeError, KeyError):
        return None


def _try_xml_format(text: str) -> list[ToolCall] | None:
    if "<tool_call>" not in text:
        return None
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not matches:
        return None
    calls = []
    for match in matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "name" in data:
                args = data.get("arguments", data.get("parameters", {}))
                calls.append(
                    ToolCall(
                        id=generate_tool_call_id(),
                        type="function",
                        function=FunctionCall(
                            name=data["name"],
                            arguments=serialize_tool_call_arguments(args),
                        ),
                    )
                )
        except (json.JSONDecodeError, KeyError):
            continue
    return calls if calls else None
