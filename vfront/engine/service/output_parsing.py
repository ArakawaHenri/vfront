"""Engine-side output parsing sessions for reasoning and tool extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

from vfront.engine.service.tokenizer import build_chat_template_kwargs
from vfront.engine.service.tool_parsing import build_parser_request, map_vllm_delta_message
from vfront.shared.config.models import ReasoningConfig
from vfront.shared.engine.reasoning import ReasoningEffort
from vfront.shared.engine.types import ParsedToolCall, SemanticOutputDelta


@dataclass(slots=True, frozen=True)
class ParsedOutputState:
    text: str
    reasoning_text: str | None = None
    semantic: SemanticOutputDelta | None = None


@dataclass(slots=True)
class EngineOutputParserSession:
    """Per-request parser state for engine-side reasoning/tool parsing."""

    parser_request: Any
    reasoning_parsers: list[Any | None] = field(default_factory=list)
    tool_parsers: list[Any | None] = field(default_factory=list)
    previous_content_texts: list[str] = field(default_factory=list)
    previous_content_token_ids: list[tuple[int, ...]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        tokenizer: Any,
        num_choices: int,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        tool_parser_name: str | None,
        reasoning_parser_name: str | None,
        reasoning_effort: ReasoningEffort | None,
        reasoning_config: ReasoningConfig | None,
    ) -> EngineOutputParserSession:
        parser_request = build_parser_request(tools=tools, tool_choice=tool_choice)
        chat_template_kwargs = build_chat_template_kwargs(
            tokenizer=tokenizer,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            reasoning_effort=reasoning_effort,
            reasoning_config=reasoning_config,
        )

        reasoning_parser_cls = (
            ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
            if reasoning_parser_name is not None
            else None
        )
        tool_parser_cls = (
            ToolParserManager.get_tool_parser(tool_parser_name)
            if tool_parser_name is not None
            else None
        )

        return cls(
            parser_request=parser_request,
            reasoning_parsers=[
                (
                    reasoning_parser_cls(
                        tokenizer,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    if reasoning_parser_cls is not None
                    else None
                )
                for _ in range(num_choices)
            ],
            tool_parsers=[
                tool_parser_cls(tokenizer) if tool_parser_cls is not None else None
                for _ in range(num_choices)
            ],
            previous_content_texts=["" for _ in range(num_choices)],
            previous_content_token_ids=[() for _ in range(num_choices)],
        )

    def parse(
        self,
        *,
        choice_index: int,
        current_text: str,
        current_token_ids: tuple[int, ...],
    ) -> ParsedOutputState:
        reasoning_parser = self.reasoning_parsers[choice_index]
        tool_parser = self.tool_parsers[choice_index]

        parsed_reasoning: str | None = None
        parsed_content = current_text
        parsed_content_token_ids = current_token_ids

        if reasoning_parser is not None:
            parsed_reasoning, content = reasoning_parser.extract_reasoning(
                current_text,
                self.parser_request,
            )
            parsed_content = content or ""
            parsed_content_token_ids = tuple(
                reasoning_parser.extract_content_ids(list(current_token_ids))
            )

        semantic = None
        if tool_parser is not None:
            previous_content = self.previous_content_texts[choice_index]
            previous_content_token_ids = self.previous_content_token_ids[choice_index]

            if parsed_content.startswith(previous_content):
                content_delta = parsed_content[len(previous_content) :]
            else:
                previous_content = ""
                previous_content_token_ids = ()
                content_delta = parsed_content

            if len(parsed_content_token_ids) >= len(previous_content_token_ids):
                delta_content_token_ids = parsed_content_token_ids[
                    len(previous_content_token_ids) :
                ]
            else:
                previous_content_token_ids = ()
                delta_content_token_ids = parsed_content_token_ids

            if content_delta or previous_content != parsed_content:
                delta_message = tool_parser.extract_tool_calls_streaming(
                    previous_text=previous_content,
                    current_text=parsed_content,
                    delta_text=content_delta,
                    previous_token_ids=previous_content_token_ids,
                    current_token_ids=parsed_content_token_ids,
                    delta_token_ids=delta_content_token_ids,
                    request=self.parser_request,
                )
                semantic = map_vllm_delta_message(delta_message)

            self.previous_content_texts[choice_index] = parsed_content
            self.previous_content_token_ids[choice_index] = parsed_content_token_ids

        return ParsedOutputState(
            text=parsed_content,
            reasoning_text=parsed_reasoning,
            semantic=semantic,
        )

    def parse_final(
        self,
        *,
        choice_index: int,
        current_text: str,
    ) -> tuple[ParsedToolCall, ...] | None:
        tool_parser = self.tool_parsers[choice_index]
        if tool_parser is None:
            return None

        reasoning_parser = self.reasoning_parsers[choice_index]
        content = current_text
        if reasoning_parser is not None:
            _, parsed_content = reasoning_parser.extract_reasoning(
                current_text,
                self.parser_request,
            )
            content = parsed_content or ""

        info = tool_parser.extract_tool_calls(content, self.parser_request)
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
