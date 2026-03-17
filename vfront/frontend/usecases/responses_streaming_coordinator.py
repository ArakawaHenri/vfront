"""Streaming coordination with persistence for the Responses API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from vfront.adapter.response import (
    _response_to_history_messages,
    messages_for_response_continuation,
)
from vfront.adapter.response_streaming import build_response_failure_events
from vfront.frontend.usecases.responses_streaming_agentic import (
    PreparedStreamingAgenticResponse,
    StreamingAgenticCallbacks,
    stream_agentic_response,
)
from vfront.frontend.usecases.responses_streaming_direct import (
    PreparedDirectStreamingResponse,
    StreamingDirectCallbacks,
    stream_direct_response,
)
from vfront.protocol.response import ResponseObject

logger = logging.getLogger(__name__)

PersistResponseState = Callable[
    [ResponseObject, list[dict[str, Any]], list[dict[str, Any]]],
    Awaitable[None],
]
BuildFailedResponse = Callable[..., ResponseObject]
ExtractSequenceNumber = Callable[[str], int | None]
ExtractResponsePayload = Callable[[str], dict[str, Any] | None]
ExecuteMcpToolCall = Callable[..., Awaitable[Any]]
TokenizeResponseMessages = Callable[..., Awaitable[list[int]]]


@dataclass(slots=True)
class PreparedPersistedStreamingResponse:
    response_id: str
    created_at: float
    request: Any
    all_tools: list[Any] | None
    conversation_messages: list[dict[str, Any]]
    stored_input_items: list[dict[str, Any]]
    prepared_generation: Any
    runtime: Any
    mcp_manager: Any
    preflight_prompt_token_ids: list[int] | None
    preflight_exception: Exception | None
    use_agentic_tools: bool
    persist_response: bool
    max_tool_rounds: int
    timeout_seconds: int


@dataclass(slots=True)
class PersistedStreamingCallbacks:
    tokenize_response_messages: TokenizeResponseMessages
    execute_mcp_tool_call: ExecuteMcpToolCall
    persist_response_state: PersistResponseState
    build_failed_response: BuildFailedResponse
    extract_sequence_number: ExtractSequenceNumber
    extract_response_payload: ExtractResponsePayload


async def stream_response_with_persistence(
    *,
    prepared: PreparedPersistedStreamingResponse,
    callbacks: PersistedStreamingCallbacks,
) -> AsyncIterator[str]:
    """Stream a response while persisting the final accumulated state when enabled."""
    accumulated_response: dict[str, Any] | None = None
    stored_context_messages = prepared.conversation_messages
    response_persisted = False

    async def _persist_accumulated_response() -> None:
        nonlocal response_persisted
        if response_persisted or not prepared.persist_response or accumulated_response is None:
            return

        response = ResponseObject.model_validate(accumulated_response)
        context_messages = messages_for_response_continuation(stored_context_messages)
        if not prepared.use_agentic_tools:
            context_messages = messages_for_response_continuation(
                prepared.conversation_messages
            ) + _response_to_history_messages(
                response,
                include_instructions=False,
            )
        await callbacks.persist_response_state(
            response,
            context_messages,
            prepared.stored_input_items,
        )
        response_persisted = True

    if prepared.preflight_exception is not None:
        logger.exception("Error during response streaming preflight")
        accumulated_response = callbacks.build_failed_response(
            response_id=prepared.response_id,
            created_at=prepared.created_at,
            request=prepared.request,
            code="stream_error",
            message=str(prepared.preflight_exception),
        ).model_dump()
        await _persist_accumulated_response()
        failure_events, _ = build_response_failure_events(
            request=prepared.request,
            message=str(prepared.preflight_exception),
            response_id=prepared.response_id,
            created_at=prepared.created_at,
            initial_sequence_number=0,
            code="stream_error",
        )
        for event in failure_events:
            yield event
        return

    if prepared.use_agentic_tools:
        def _set_accumulated_response(value: dict[str, Any]) -> None:
            nonlocal accumulated_response
            accumulated_response = value

        def _set_stored_context_messages(value: list[dict[str, Any]]) -> None:
            nonlocal stored_context_messages
            stored_context_messages = value

        async for event in stream_agentic_response(
            prepared=PreparedStreamingAgenticResponse(
                response_id=prepared.response_id,
                created_at=prepared.created_at,
                request=prepared.request,
                all_tools=prepared.all_tools,
                conversation_messages=prepared.conversation_messages,
                prepared_generation=prepared.prepared_generation,
                runtime=prepared.runtime,
                mcp_manager=prepared.mcp_manager,
                preflight_prompt_token_ids=prepared.preflight_prompt_token_ids,
                max_tool_rounds=prepared.max_tool_rounds,
                timeout_seconds=prepared.timeout_seconds,
            ),
            callbacks=StreamingAgenticCallbacks(
                tokenize_response_messages=callbacks.tokenize_response_messages,
                execute_mcp_tool_call=callbacks.execute_mcp_tool_call,
                persist_accumulated_response=_persist_accumulated_response,
                build_failed_response=callbacks.build_failed_response,
                extract_sequence_number=callbacks.extract_sequence_number,
                extract_response_payload=callbacks.extract_response_payload,
                set_accumulated_response=_set_accumulated_response,
                set_stored_context_messages=_set_stored_context_messages,
            ),
        ):
            yield event
    else:
        def _set_accumulated_response(value: dict[str, Any]) -> None:
            nonlocal accumulated_response
            accumulated_response = value

        async for event in stream_direct_response(
            prepared=PreparedDirectStreamingResponse(
                response_id=prepared.response_id,
                created_at=prepared.created_at,
                request=prepared.request,
                prepared_generation=prepared.prepared_generation,
                runtime=prepared.runtime,
                preflight_prompt_token_ids=prepared.preflight_prompt_token_ids,
            ),
            callbacks=StreamingDirectCallbacks(
                tokenize_response_messages=callbacks.tokenize_response_messages,
                persist_accumulated_response=_persist_accumulated_response,
                build_failed_response=callbacks.build_failed_response,
                extract_sequence_number=callbacks.extract_sequence_number,
                extract_response_payload=callbacks.extract_response_payload,
                set_accumulated_response=_set_accumulated_response,
            ),
        ):
            yield event

    await _persist_accumulated_response()
