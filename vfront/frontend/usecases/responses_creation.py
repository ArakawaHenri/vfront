"""Create-response orchestration helpers for the Responses API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from vfront.adapter.response import messages_for_response_continuation
from vfront.protocol.job import JobResolvedTarget, ResponseExecution, ResponseJobSpec
from vfront.protocol.response import ResponseObject

TokenizeResponseMessages = Callable[..., Awaitable[list[int]]]
PersistResponseState = Callable[..., Awaitable[None]]
ExecuteResponseGeneration = Callable[..., Awaitable[Any]]
BuildResponseFromGeneration = Callable[..., ResponseObject]


@dataclass(slots=True)
class PreparedResponseCreation:
    request: Any
    effective_request: Any
    all_tools: list[Any] | None
    conversation_messages: list[dict[str, Any]]
    stored_input_items: list[dict[str, Any]]
    prepared_generation: Any
    resolved_target: Any
    runtime: Any
    response_id: str
    created_at: float
    use_agentic_tools: bool
    persist_response: bool
    mcp_settings: Any


async def create_background_response_use_case(
    *,
    prepared: PreparedResponseCreation,
    requested_model: str,
    job_store: Any,
    runner: Any,
    tokenize_response_messages: TokenizeResponseMessages,
    build_queued_response: Callable[..., ResponseObject],
    build_response_job_spec: Callable[..., ResponseJobSpec],
    persist_response_state: PersistResponseState,
) -> ResponseObject:
    await tokenize_response_messages(
        prepared.runtime,
        prepared.prepared_generation.generation_messages,
        reasoning_effort=prepared.prepared_generation.params.reasoning_effort,
    )
    target = JobResolvedTarget(
        requested_model=prepared.resolved_target.requested_model,
        public_name=prepared.resolved_target.public_name,
        runtime_model=prepared.resolved_target.runtime_model,
        backend=prepared.resolved_target.backend,
        adapter=prepared.resolved_target.adapter,
        tool_parser=prepared.resolved_target.tool_parser,
    )
    queued_response = build_queued_response(
        response_id=prepared.response_id,
        created_at=prepared.created_at,
        request=prepared.effective_request,
    )
    job_spec = build_response_job_spec(
        response_id=prepared.response_id,
        created_at=prepared.created_at,
        request=prepared.effective_request,
        target=target,
        conversation_messages=prepared.conversation_messages,
    )
    execution = ResponseExecution(
        response_id=prepared.response_id,
        status="queued",
        requested_model=requested_model,
        target=target,
        created_at=prepared.created_at,
    )

    if prepared.persist_response:
        await persist_response_state(
            queued_response,
            input_items=prepared.stored_input_items,
        )
    await job_store.put_response_spec(job_spec)
    await job_store.put_response_execution(execution)
    await runner.notify()
    return queued_response


async def create_non_streaming_response_use_case(
    *,
    prepared: PreparedResponseCreation,
    mcp_manager: Any,
    execute_response_generation: ExecuteResponseGeneration,
    build_response_from_generation: BuildResponseFromGeneration,
    persist_response_state: PersistResponseState,
) -> ResponseObject:
    generation_result = await execute_response_generation(
        conversation_messages=prepared.conversation_messages,
        generation_messages=prepared.prepared_generation.generation_messages,
        tools=prepared.all_tools,
        parallel_tool_calls=prepared.effective_request.parallel_tool_calls,
        max_tool_calls=prepared.effective_request.max_tool_calls,
        runtime=prepared.runtime,
        params=prepared.prepared_generation.params,
        response_id=prepared.response_id,
        use_agentic_tools=prepared.use_agentic_tools,
        mcp_manager=mcp_manager,
        max_tool_rounds=prepared.mcp_settings.max_tool_rounds,
        timeout=prepared.mcp_settings.timeout_seconds,
    )
    response = build_response_from_generation(
        engine_output=generation_result.final_output,
        request=prepared.effective_request,
        response_id=prepared.response_id,
        created_at=prepared.created_at,
        tool_executions=generation_result.tool_executions,
        pending_client_calls=generation_result.pending_client_calls,
    )

    if prepared.persist_response:
        await persist_response_state(
            response=response,
            context_messages=messages_for_response_continuation(
                generation_result.final_messages
            ),
            input_items=prepared.stored_input_items,
        )
    return response
