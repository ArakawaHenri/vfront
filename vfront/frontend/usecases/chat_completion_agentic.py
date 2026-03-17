"""Agentic chat-completion orchestration shared by chat route handlers."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from vfront.adapter.streaming import stream_and_collect
from vfront.adapter.tool_calls import build_tool_calls_from_parsed, build_tool_calls_from_text
from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.mcp.settings import MCPSettings
from vfront.protocol.common import ToolCall
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.generation import collect_final_stream_output
from vfront.shared.engine.reasoning import ReasoningEffort
from vfront.shared.engine.tokenization import tokenize_chat_messages
from vfront.shared.engine.types import CompletionDelta, GenerateOutput, GenerateParams


@dataclass(slots=True)
class PreparedChatAgenticRequest:
    request_id: str
    model: str
    n: int
    reasoning_effort: ReasoningEffort | None
    prompt_token_ids: list[int]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    params: GenerateParams
    runtime: EngineClient
    adapter: str | None
    system_fingerprint: str | None
    mcp_manager: MCPClientManager | None
    mcp_settings: MCPSettings


def detect_tool_calls_from_completion(completion: CompletionDelta) -> list[ToolCall] | None:
    if completion.parsed_tool_calls is not None:
        return build_tool_calls_from_parsed(completion.parsed_tool_calls)
    if completion.text:
        return build_tool_calls_from_text(completion.text)
    return None


async def stream_agentic_chat_completion(
    *,
    prepared: PreparedChatAgenticRequest,
    execute_mcp_tool_call_fn: Callable[..., Awaitable[Any]],
) -> AsyncIterator[str]:
    round_num = 0
    while round_num <= prepared.mcp_settings.max_tool_rounds:
        round_num += 1
        rid = prepared.request_id if round_num == 1 else f"{prepared.request_id}-r{round_num}"
        generator = cast(
            "AsyncGenerator[GenerateOutput, None]",
            prepared.runtime.generate(prepared.prompt_token_ids, prepared.params, rid),
        )
        accumulated: list[str] = []
        collected_tool_calls: list[ToolCall] = []

        async for event in stream_and_collect(
            request_id=rid,
            model=prepared.model,
            output_generator=generator,
            n=prepared.n,
            has_tool_definitions=bool(prepared.tools),
            accumulated=accumulated,
            collected_tool_calls=collected_tool_calls,
            lora_name=prepared.adapter,
            system_fingerprint=prepared.system_fingerprint,
        ):
            yield event

        detected: list[ToolCall] | None = collected_tool_calls
        if not detected:
            accumulated_text = accumulated[0] if accumulated else ""
            detected = (
                build_tool_calls_from_text(accumulated_text)
                if accumulated_text and prepared.tools
                else []
            )
        mcp_tool_calls = _mcp_tool_calls(prepared=prepared, detected_tool_calls=detected)
        if not mcp_tool_calls:
            break

        await _append_agentic_round_messages(
            prepared=prepared,
            detected_tool_calls=detected or [],
            mcp_tool_calls=mcp_tool_calls,
            execute_mcp_tool_call_fn=execute_mcp_tool_call_fn,
        )

    yield "data: [DONE]\n\n"


async def run_non_streaming_agentic_chat_completion(
    *,
    prepared: PreparedChatAgenticRequest,
    initial_output: GenerateOutput,
    execute_mcp_tool_call_fn: Callable[..., Awaitable[Any]],
) -> GenerateOutput:
    final_output = initial_output
    if not final_output.outputs:
        return final_output

    detected_tool_calls = (
        detect_tool_calls_from_completion(final_output.outputs[0]) if prepared.tools else []
    )
    mcp_tool_calls = _mcp_tool_calls(
        prepared=prepared,
        detected_tool_calls=detected_tool_calls,
    )

    rounds = 0
    while mcp_tool_calls and rounds < prepared.mcp_settings.max_tool_rounds:
        rounds += 1
        await _append_agentic_round_messages(
            prepared=prepared,
            detected_tool_calls=detected_tool_calls or [],
            mcp_tool_calls=mcp_tool_calls,
            execute_mcp_tool_call_fn=execute_mcp_tool_call_fn,
        )

        final_output = await collect_final_stream_output(
            prepared.runtime.generate(
                prepared.prompt_token_ids,
                prepared.params,
                f"{prepared.request_id}-r{rounds}",
            ),
            empty_error_message="Engine returned no output in agentic round",
        )
        if not final_output.outputs:
            break

        detected_tool_calls = (
            detect_tool_calls_from_completion(final_output.outputs[0]) if prepared.tools else []
        )
        mcp_tool_calls = _mcp_tool_calls(
            prepared=prepared,
            detected_tool_calls=detected_tool_calls,
        )

    return final_output


def _mcp_tool_calls(
    *,
    prepared: PreparedChatAgenticRequest,
    detected_tool_calls: list[ToolCall] | None,
) -> list[ToolCall]:
    if prepared.mcp_manager is None:
        return []
    return [
        tool_call
        for tool_call in (detected_tool_calls or [])
        if prepared.mcp_manager.is_mcp_tool(tool_call.function.name)
    ]


async def _append_agentic_round_messages(
    *,
    prepared: PreparedChatAgenticRequest,
    detected_tool_calls: list[ToolCall],
    mcp_tool_calls: list[ToolCall],
    execute_mcp_tool_call_fn: Callable[..., Awaitable[Any]],
) -> None:
    if prepared.mcp_manager is None:
        raise RuntimeError("MCP manager is required for agentic chat execution.")
    prepared.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call.model_dump() for tool_call in detected_tool_calls],
        }
    )
    for tool_call in mcp_tool_calls:
        execution = await execute_mcp_tool_call_fn(
            prepared.mcp_manager,
            tool_call,
            timeout=prepared.mcp_settings.timeout_seconds,
        )
        prepared.messages.append(
            {
                "role": "tool",
                "tool_call_id": execution.tool_call_id,
                "content": execution.result_text,
            }
        )

    try:
        prepared.prompt_token_ids = await tokenize_chat_messages(
            prepared.runtime,
            prepared.messages,
            tools=prepared.tools,
            reasoning_effort=prepared.reasoning_effort,
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="reasoning_effort") from exc
