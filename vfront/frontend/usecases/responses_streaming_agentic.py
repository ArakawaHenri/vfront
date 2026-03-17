"""Agentic streaming orchestration for the Responses API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from vfront.adapter.assistant_output import normalize_assistant_output
from vfront.adapter.response import build_generation_messages
from vfront.adapter.response_streaming import (
    build_function_tool_call_events,
    build_response_failure_events,
    build_response_prelude_events,
    format_response_completed_event,
    generate_response_stream_events,
)
from vfront.adapter.tool_calls import limit_parallel_tool_calls
from vfront.frontend.api.v1.helper import build_response_from_generation
from vfront.frontend.api.v1.helper.response_execution import PreparedResponseGeneration
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.mcp.executor import append_final_output_to_messages
from vfront.protocol.common import ToolCall
from vfront.protocol.response import FunctionToolCall, FunctionToolDefinition, ResponseCreateRequest
from vfront.shared.engine.client import EngineClient
from vfront.shared.engine.generation import collect_final_stream_output
from vfront.shared.engine.types import GenerateOutput

logger = logging.getLogger(__name__)

TokenizeResponseMessages = Callable[..., Awaitable[list[int]]]
PersistAccumulatedResponse = Callable[[], Awaitable[None]]
ExecuteMcpToolCall = Callable[..., Awaitable[Any]]
BuildFailedResponse = Callable[..., Any]
ExtractSequenceNumber = Callable[[str], int | None]
ExtractResponsePayload = Callable[[str], dict[str, Any] | None]
SetAccumulatedResponse = Callable[[dict[str, Any]], None]
SetStoredContextMessages = Callable[[list[dict[str, Any]]], None]


@dataclass(slots=True)
class PreparedStreamingAgenticResponse:
    """Prepared route state required for agentic response streaming."""

    response_id: str
    created_at: float
    request: ResponseCreateRequest
    all_tools: list[FunctionToolDefinition] | None
    conversation_messages: list[dict[str, Any]]
    prepared_generation: PreparedResponseGeneration
    runtime: EngineClient
    mcp_manager: MCPClientManager
    preflight_prompt_token_ids: list[int] | None
    max_tool_rounds: int
    timeout_seconds: int


@dataclass(slots=True)
class StreamingAgenticCallbacks:
    """Route-provided callbacks for patch-sensitive helpers and persistence."""

    tokenize_response_messages: TokenizeResponseMessages
    execute_mcp_tool_call: ExecuteMcpToolCall
    persist_accumulated_response: PersistAccumulatedResponse
    build_failed_response: BuildFailedResponse
    extract_sequence_number: ExtractSequenceNumber
    extract_response_payload: ExtractResponsePayload
    set_accumulated_response: SetAccumulatedResponse
    set_stored_context_messages: SetStoredContextMessages


async def _single_output_stream(output: GenerateOutput) -> AsyncIterator[GenerateOutput]:
    yield output


async def stream_agentic_response(
    *,
    prepared: PreparedStreamingAgenticResponse,
    callbacks: StreamingAgenticCallbacks,
) -> AsyncIterator[str]:
    """Yield response stream events for the agentic MCP-tool branch."""
    prelude_events, seq = build_response_prelude_events(
        request=prepared.request,
        response_id=prepared.response_id,
        created_at=prepared.created_at,
    )
    for event in prelude_events:
        yield event

    current_messages = deepcopy(prepared.conversation_messages)
    executed_tool_items: list[FunctionToolCall] = []
    final_output: GenerateOutput | None = None
    final_prompt_tokens = 0
    rounds = 0

    try:
        while rounds < prepared.max_tool_rounds:
            rounds += 1
            if rounds == 1 and prepared.preflight_prompt_token_ids is not None:
                prompt_token_ids = prepared.preflight_prompt_token_ids
            else:
                prompt_token_ids = await callbacks.tokenize_response_messages(
                    prepared.runtime,
                    build_generation_messages(current_messages, prepared.all_tools),
                    reasoning_effort=prepared.prepared_generation.params.reasoning_effort,
                )
            final_prompt_tokens = len(prompt_token_ids)
            round_output = await collect_final_stream_output(
                prepared.runtime.generate(
                    prompt_token_ids,
                    deepcopy(prepared.prepared_generation.params),
                    f"{prepared.response_id}-{rounds}",
                ),
                empty_error_message="Engine returned no output",
            )

            final_output = round_output
            generated_text = round_output.outputs[0].text if round_output.outputs else ""
            normalized = normalize_assistant_output(
                generated_text,
                has_tool_definitions=bool(prepared.all_tools),
            )
            detected_calls = limit_parallel_tool_calls(
                list(normalized.tool_calls or []),
                parallel_tool_calls=prepared.request.parallel_tool_calls,
            )
            client_calls = [
                tool_call
                for tool_call in detected_calls
                if not prepared.mcp_manager.is_mcp_tool(tool_call.function.name)
            ]
            mcp_calls = [
                tool_call
                for tool_call in detected_calls
                if prepared.mcp_manager.is_mcp_tool(tool_call.function.name)
            ]

            if not mcp_calls:
                if client_calls:
                    current_messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call.model_dump() for tool_call in detected_calls],
                        }
                    )
                    for idx, tool_call in enumerate(client_calls):
                        tool_item = _tool_item_from_call(tool_call)
                        tool_events, seq = build_function_tool_call_events(
                            tool_item=tool_item,
                            output_index=len(executed_tool_items) + idx,
                            initial_sequence_number=seq,
                        )
                        for event in tool_events:
                            yield event

                    mixed_response = build_response_from_generation(
                        engine_output=round_output,
                        request=prepared.request,
                        response_id=prepared.response_id,
                        created_at=prepared.created_at,
                        pending_client_calls=client_calls,
                    )
                    callbacks.set_accumulated_response(mixed_response.model_dump())
                    callbacks.set_stored_context_messages(current_messages)
                    await callbacks.persist_accumulated_response()
                    yield format_response_completed_event(
                        mixed_response,
                        sequence_number=seq,
                    )
                    return
                break

            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call.model_dump() for tool_call in detected_calls],
                }
            )
            emit_mcp_events = not client_calls
            for tool_call in mcp_calls:
                tool_item = _tool_item_from_call(tool_call)
                if emit_mcp_events:
                    tool_events, seq = build_function_tool_call_events(
                        tool_item=tool_item,
                        output_index=len(executed_tool_items),
                        initial_sequence_number=seq,
                    )
                    for event in tool_events:
                        yield event
                    executed_tool_items.append(tool_item)

                execution = await callbacks.execute_mcp_tool_call(
                    prepared.mcp_manager,
                    tool_call,
                    timeout=prepared.timeout_seconds,
                )
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": execution.tool_call_id,
                        "content": execution.result_text,
                    }
                )

            if client_calls:
                for idx, tool_call in enumerate(client_calls):
                    tool_item = _tool_item_from_call(tool_call)
                    tool_events, seq = build_function_tool_call_events(
                        tool_item=tool_item,
                        output_index=len(executed_tool_items) + idx,
                        initial_sequence_number=seq,
                    )
                    for event in tool_events:
                        yield event

                mixed_response = build_response_from_generation(
                    engine_output=round_output,
                    request=prepared.request,
                    response_id=prepared.response_id,
                    created_at=prepared.created_at,
                    pending_client_calls=client_calls,
                )
                callbacks.set_accumulated_response(mixed_response.model_dump())
                callbacks.set_stored_context_messages(current_messages)
                await callbacks.persist_accumulated_response()
                yield format_response_completed_event(
                    mixed_response,
                    sequence_number=seq,
                )
                return
        else:
            prompt_token_ids = await callbacks.tokenize_response_messages(
                prepared.runtime,
                build_generation_messages(current_messages, prepared.all_tools),
                reasoning_effort=prepared.prepared_generation.params.reasoning_effort,
            )
            final_prompt_tokens = len(prompt_token_ids)
            final_output = await collect_final_stream_output(
                prepared.runtime.generate(
                    prompt_token_ids,
                    deepcopy(prepared.prepared_generation.params),
                    f"{prepared.response_id}-final",
                ),
                empty_error_message="Engine returned no output",
            )

        assert final_output is not None
        final_context_messages = append_final_output_to_messages(
            current_messages,
            final_output,
            has_tool_definitions=bool(prepared.all_tools),
            parallel_tool_calls=prepared.request.parallel_tool_calls,
        )

        async for event in generate_response_stream_events(
            engine_stream=_single_output_stream(final_output),
            request=prepared.request,
            response_id=prepared.response_id,
            created_at=prepared.created_at,
            prompt_tokens=final_prompt_tokens,
            prefix_tool_calls=executed_tool_items,
            initial_sequence_number=seq,
            emit_prelude=False,
            emit_prefix_tool_call_events=False,
        ):
            event_seq = callbacks.extract_sequence_number(event)
            if event_seq is not None:
                seq = event_seq + 1
            event_response = callbacks.extract_response_payload(event)
            if event_response is not None:
                callbacks.set_accumulated_response(event_response)
                callbacks.set_stored_context_messages(final_context_messages)
                await callbacks.persist_accumulated_response()
            yield event
        callbacks.set_stored_context_messages(final_context_messages)
    except Exception as exc:
        logger.exception("Error during agentic response streaming")
        callbacks.set_accumulated_response(
            callbacks.build_failed_response(
                response_id=prepared.response_id,
                created_at=prepared.created_at,
                request=prepared.request,
                code="stream_error",
                message=str(exc),
            ).model_dump()
        )
        callbacks.set_stored_context_messages(current_messages)
        await callbacks.persist_accumulated_response()
        failure_events, _ = build_response_failure_events(
            request=prepared.request,
            message=str(exc),
            response_id=prepared.response_id,
            created_at=prepared.created_at,
            initial_sequence_number=seq,
            code="stream_error",
        )
        for event in failure_events:
            yield event


def _tool_item_from_call(tool_call: ToolCall) -> FunctionToolCall:
    return FunctionToolCall(
        name=tool_call.function.name,
        arguments=tool_call.function.arguments,
        call_id=tool_call.id,
        status="completed",
    )
