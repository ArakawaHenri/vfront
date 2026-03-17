"""Direct streaming orchestration for the Responses API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from vfront.adapter.response_streaming import (
    build_response_failure_events,
    generate_response_stream_events,
)
from vfront.frontend.api.v1.helper.response_execution import PreparedResponseGeneration
from vfront.protocol.response import ResponseCreateRequest, ResponseObject
from vfront.shared.engine.client import EngineClient

logger = logging.getLogger(__name__)

TokenizeResponseMessages = Callable[..., Awaitable[list[int]]]
PersistAccumulatedResponse = Callable[[], Awaitable[None]]
BuildFailedResponse = Callable[..., ResponseObject]
ExtractSequenceNumber = Callable[[str], int | None]
ExtractResponsePayload = Callable[[str], dict[str, Any] | None]
SetAccumulatedResponse = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PreparedDirectStreamingResponse:
    """Prepared route state required for non-agentic response streaming."""

    response_id: str
    created_at: float
    request: ResponseCreateRequest
    prepared_generation: PreparedResponseGeneration
    runtime: EngineClient
    preflight_prompt_token_ids: list[int] | None


@dataclass(slots=True)
class StreamingDirectCallbacks:
    """Route-provided callbacks for patch-sensitive helpers and persistence."""

    tokenize_response_messages: TokenizeResponseMessages
    persist_accumulated_response: PersistAccumulatedResponse
    build_failed_response: BuildFailedResponse
    extract_sequence_number: ExtractSequenceNumber
    extract_response_payload: ExtractResponsePayload
    set_accumulated_response: SetAccumulatedResponse


async def stream_direct_response(
    *,
    prepared: PreparedDirectStreamingResponse,
    callbacks: StreamingDirectCallbacks,
) -> AsyncIterator[str]:
    """Yield response stream events for the direct non-agentic branch."""
    seq = 0
    try:
        prompt_token_ids = prepared.preflight_prompt_token_ids
        if prompt_token_ids is None:
            prompt_token_ids = await callbacks.tokenize_response_messages(
                prepared.runtime,
                prepared.prepared_generation.generation_messages,
                reasoning_effort=prepared.prepared_generation.params.reasoning_effort,
            )
        generator = prepared.runtime.generate(
            prompt_token_ids,
            prepared.prepared_generation.params,
            prepared.response_id,
        )
        async for event in generate_response_stream_events(
            engine_stream=generator,
            request=prepared.request,
            response_id=prepared.response_id,
            created_at=prepared.created_at,
            prompt_tokens=len(prompt_token_ids),
        ):
            event_seq = callbacks.extract_sequence_number(event)
            if event_seq is not None:
                seq = event_seq + 1
            event_response = callbacks.extract_response_payload(event)
            if event_response is not None:
                callbacks.set_accumulated_response(event_response)
                await callbacks.persist_accumulated_response()
            yield event
    except Exception as exc:
        logger.exception("Error during response streaming")
        callbacks.set_accumulated_response(
            callbacks.build_failed_response(
                response_id=prepared.response_id,
                created_at=prepared.created_at,
                request=prepared.request,
                code="stream_error",
                message=str(exc),
            ).model_dump()
        )
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
