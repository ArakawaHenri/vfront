"""Engine-side semantic tool-call parsing backed directly by vLLM parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager

from vfront.shared.engine.types import (
    ParsedToolCall,
    SemanticOutputDelta,
    SemanticToolCallDelta,
)


def list_available_tool_parsers() -> list[str]:
    """Return parser names exposed by the installed vLLM package."""
    return sorted(ToolParserManager.list_registered())


def parser_exists(name: str) -> bool:
    return name in ToolParserManager.list_registered()


def resolve_effective_tool_parser(
    configured_tool_parser: str | None,
    *,
    backend_model: str,
    available_tool_parsers: list[str] | None = None,
) -> str | None:
    """Resolve the effective parser using explicit configuration."""
    available = set(available_tool_parsers or list_available_tool_parsers())

    if configured_tool_parser is None:
        return None

    if configured_tool_parser == "auto":
        raise ValueError(
            "The 'auto' tool parser resolution is no longer supported. "
            f"Please explicitly configure a specific tool parser for model '{backend_model}'. "
            f"Available parsers: {sorted(available)}"
        )

    if configured_tool_parser not in available:
        raise ValueError(
            f"Configured tool parser '{configured_tool_parser}' is not available. "
            f"Available parsers: {sorted(available)}"
        )

    return configured_tool_parser


def build_parser_request(
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> ChatCompletionRequest:
    """Build a minimal vLLM ChatCompletionRequest for parser consumption."""
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": ""}],
        "model": "compat-parser",
        "stream": True,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    return ChatCompletionRequest.model_validate(payload)


def map_vllm_delta_message(delta_message: Any | None) -> SemanticOutputDelta | None:
    if delta_message is None:
        return None
    tool_call_deltas = tuple(
        SemanticToolCallDelta(
            index=tool_delta.index,
            id=tool_delta.id,
            type=tool_delta.type,
            name=tool_delta.function.name if tool_delta.function is not None else None,
            arguments_delta=(
                tool_delta.function.arguments if tool_delta.function is not None else None
            ),
        )
        for tool_delta in getattr(delta_message, "tool_calls", []) or []
    )
    content_delta = getattr(delta_message, "content", None)
    if content_delta is None and not tool_call_deltas:
        return None
    return SemanticOutputDelta(
        content_delta=content_delta,
        tool_call_deltas=tool_call_deltas,
    )


@dataclass(slots=True)
class EngineToolParserSession:
    """Per-request parser state for engine-side semantic tool parsing."""

    parser_name: str
    tokenizer: Any
    parser_request: ChatCompletionRequest
    parser_cls: Any
    parser_instances: list[Any] = field(default_factory=list)
    previous_texts: list[str] = field(default_factory=list)
    previous_token_ids: list[tuple[int, ...]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        parser_name: str,
        tokenizer: Any,
        num_choices: int,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> EngineToolParserSession:
        parser_cls = ToolParserManager.get_tool_parser(parser_name)
        parser_request = build_parser_request(tools=tools, tool_choice=tool_choice)
        return cls(
            parser_name=parser_name,
            tokenizer=tokenizer,
            parser_request=parser_request,
            parser_cls=parser_cls,
            parser_instances=[parser_cls(tokenizer) for _ in range(num_choices)],
            previous_texts=["" for _ in range(num_choices)],
            previous_token_ids=[() for _ in range(num_choices)],
        )

    def parse(
        self,
        *,
        choice_index: int,
        current_text: str,
        current_token_ids: tuple[int, ...],
    ) -> SemanticOutputDelta | None:
        previous_text = self.previous_texts[choice_index]
        previous_token_ids = self.previous_token_ids[choice_index]
        delta_text = current_text[len(previous_text) :]
        delta_token_ids = current_token_ids[len(previous_token_ids) :]
        parser = self.parser_instances[choice_index]
        delta_message = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=self.parser_request,
        )
        self.previous_texts[choice_index] = current_text
        self.previous_token_ids[choice_index] = current_token_ids
        return map_vllm_delta_message(delta_message)

    def parse_final(
        self,
        *,
        current_text: str,
    ) -> tuple[ParsedToolCall, ...] | None:
        parser = self.parser_cls(self.tokenizer)
        info = parser.extract_tool_calls(current_text, self.parser_request)
        if not info or not info.tools_called:
            return None
        return tuple(
            ParsedToolCall(
                id=tool_call.id,
                type=tool_call.type,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in info.tool_calls
        )
