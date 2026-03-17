"""Responses API endpoints.

Implements the OpenAI Responses API with:
- Non-streaming and streaming response creation
- Response storage, retrieval, and deletion
- Background mode with cancel support
- Multi-turn via previous_response_id
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query
from fastapiex.di import Inject
from fastapiex.settings import GetSettings
from starlette.responses import StreamingResponse

from vfront.adapter.response import (
    _conversation_messages_to_input_items,
    messages_for_response_continuation,
)
from vfront.adapter.response import (
    input_to_messages as _input_to_messages,
)
from vfront.frontend.api.v1.helper import (
    build_response_from_generation,
    execute_response_generation,
    prepare_response_generation,
    tokenize_response_messages,
)
from vfront.frontend.compat.validators import validate_responses_request
from vfront.frontend.middleware.exceptions import InvalidRequestError, NotFoundError
from vfront.frontend.service.engine.client_initializer import (
    initialize_engine_client,
)
from vfront.frontend.service.engine.router import EngineRouter
from vfront.frontend.service.jobs.lease import JobLeaseService
from vfront.frontend.service.jobs.runner import JobRunnerService
from vfront.frontend.service.jobs.store import JobStoreService
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.mcp.executor import (
    execute_mcp_tool_call,
)
from vfront.frontend.service.persistence import response_store
from vfront.frontend.service.store.lmdb import StoreService
from vfront.frontend.service.tool_parsing.service import ToolParsingService
from vfront.frontend.usecases import (
    PersistedStreamingCallbacks,
    PreparedPersistedStreamingResponse,
    PreparedResponseCreation,
    background_generate_response_use_case,
    cancel_response_use_case,
    create_background_response_use_case,
    create_non_streaming_response_use_case,
    delete_response_use_case,
    list_response_input_items_use_case,
    prepare_responses_request,
    stream_response_with_persistence,
)
from vfront.protocol.job import JobResolvedTarget, ResponseJobSpec
from vfront.protocol.response import (
    ResponseCreateRequest,
    ResponseDeleted,
    ResponseError,
    ResponseItemList,
    ResponseObject,
)
from vfront.shared.engine.errors import EngineInputError

router = APIRouter()

_STORE_NAMESPACE = response_store.RESPONSE_STORE_NAMESPACE
_RESPONSE_INPUT_ITEMS_NAMESPACE = response_store.RESPONSE_INPUT_ITEMS_NAMESPACE
_REUSABLE_PREVIOUS_RESPONSE_STATUSES = {"completed", "incomplete"}

input_to_messages = _input_to_messages


def _should_persist_response(request: ResponseCreateRequest) -> bool:
    return response_store.should_persist_response(request)


def _paginate_response_input_items(
    *,
    items: list[dict[str, Any]],
    limit: int,
    order: str,
    after: str | None,
    before: str | None,
) -> tuple[list[dict[str, Any]], bool]:
    return response_store.paginate_response_input_items(
        items=items,
        limit=limit,
        order=order,
        after=after,
        before=before,
    )


async def _store_response_state(
    store: StoreService,
    response: ResponseObject,
    context_messages: list[dict[str, Any]] | None = None,
    input_items: list[dict[str, Any]] | None = None,
) -> None:
    await response_store.store_response_state(
        store,
        response,
        context_messages=context_messages,
        input_items=input_items,
    )


async def _delete_response_state(
    store: StoreService,
    job_store: JobStoreService,
    lease_service: JobLeaseService,
    response_id: str,
) -> bool:
    return await response_store.delete_response_state(
        store,
        job_store,
        lease_service,
        response_id,
    )


async def _single_output_stream(output) -> AsyncIterator:
    yield output


def _extract_stream_event_data(event: str) -> dict[str, Any] | None:
    for line in event.split("\n"):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            return data if isinstance(data, dict) else None
    return None


def _extract_sequence_number_from_stream_event(event: str) -> int | None:
    data = _extract_stream_event_data(event)
    if data is None:
        return None
    sequence_number = data.get("sequence_number")
    return sequence_number if isinstance(sequence_number, int) else None


def _extract_response_payload_from_stream_event(event: str) -> dict[str, Any] | None:
    if "response.completed" not in event and "response.failed" not in event:
        return None
    data = _extract_stream_event_data(event)
    if data is not None:
        response = data.get("response")
        return response if isinstance(response, dict) else None
    return None


def _build_failed_response(
    *,
    response_id: str,
    created_at: float,
    request: ResponseCreateRequest,
    code: str,
    message: str,
) -> ResponseObject:
    return _build_queued_response(
        response_id=response_id,
        created_at=created_at,
        request=request,
    ).model_copy(
        update={
            "status": "failed",
            "error": ResponseError(code=code, message=message),
            "completed_at": time.time(),
        }
    )


def _build_response_job_spec(
    response_id: str,
    created_at: float,
    request: ResponseCreateRequest,
    target: JobResolvedTarget,
    conversation_messages: list[dict[str, Any]],
) -> ResponseJobSpec:
    return ResponseJobSpec(
        response_id=response_id,
        created_at=created_at,
        request=request,
        target=target,
        conversation_messages=conversation_messages,
    )


def _assert_previous_response_reusable(
    response_id: str,
    response: ResponseObject,
) -> None:
    if response.status not in _REUSABLE_PREVIOUS_RESPONSE_STATUSES:
        raise InvalidRequestError(
            f"Response '{response_id}' is not ready for continuation (status: {response.status})."
        )


def _build_queued_response(
    response_id: str,
    created_at: float,
    request: ResponseCreateRequest,
) -> ResponseObject:
    return ResponseObject(
        id=response_id,
        created_at=created_at,
        status="queued",
        model=request.model,
        output=[],
        metadata=request.metadata,
        instructions=request.instructions,
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_output_tokens,
        stop=request.stop,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        text=request.text,
        tools=request.tools,
        tool_choice=request.tool_choice,
        parallel_tool_calls=(
            True if request.parallel_tool_calls is None else request.parallel_tool_calls
        ),
        store=request.store,
        previous_response_id=request.previous_response_id,
        truncation=request.truncation,
        user=request.user,
        background=request.background,
        top_logprobs=request.top_logprobs,
        reasoning=request.reasoning,
        service_tier=request.service_tier,
        max_tool_calls=request.max_tool_calls,
    )


@router.post("/responses", response_model=ResponseObject)
async def create_response(
    request: ResponseCreateRequest,
    engine_router: EngineRouter = Inject("engine_router"),
    store: StoreService = Inject("store_service"),
    job_store: JobStoreService = Inject("job_store_service"),
    runner: JobRunnerService = Inject("job_runner_service"),
    mcp_manager: MCPClientManager = Inject("mcp_client_manager"),
    tool_parsing_service: ToolParsingService = Inject("tool_parsing_service"),
):
    validate_responses_request(request)
    prepared_request = await prepare_responses_request(
        request=request,
        engine_router=engine_router,
        store=store,
        mcp_manager=mcp_manager,
        tool_parsing_service=tool_parsing_service,
        store_namespace=_STORE_NAMESPACE,
        get_settings=GetSettings,
        validate_previous_response=_assert_previous_response_reusable,
        build_input_items=lambda response_id, messages: _conversation_messages_to_input_items(
            response_id=response_id,
            messages=messages,
        ),
        should_persist_response=_should_persist_response,
    )
    mcp_settings = GetSettings("frontend.mcp")
    effective_request = prepared_request.effective_request
    all_tools = prepared_request.all_tools
    conversation_messages = prepared_request.conversation_messages
    stored_input_items = prepared_request.stored_input_items
    prepared = prepared_request.prepared_generation
    resolved = prepared_request.initialized_client.resolved
    runtime = prepared_request.initialized_client.client
    response_id = prepared_request.response_id
    created_at = prepared_request.created_at
    use_agentic_tools = prepared_request.use_agentic_tools
    persist_response = prepared_request.persist_response
    create_prepared = PreparedResponseCreation(
        request=request,
        effective_request=effective_request,
        all_tools=all_tools,
        conversation_messages=conversation_messages,
        stored_input_items=stored_input_items,
        prepared_generation=prepared,
        resolved_target=resolved.target,
        runtime=runtime,
        response_id=response_id,
        created_at=created_at,
        use_agentic_tools=use_agentic_tools,
        persist_response=persist_response,
        mcp_settings=mcp_settings,
    )

    # Background mode
    if request.background:
        return await create_background_response_use_case(
            prepared=create_prepared,
            requested_model=request.model,
            job_store=job_store,
            runner=runner,
            tokenize_response_messages=tokenize_response_messages,
            build_queued_response=_build_queued_response,
            build_response_job_spec=_build_response_job_spec,
            persist_response_state=lambda response, **kwargs: _store_response_state(
                store,
                response,
                **kwargs,
            ),
        )

    # Streaming mode
    if request.stream:
        preflight_prompt_token_ids: list[int] | None = None
        preflight_exception: Exception | None = None
        try:
            preflight_prompt_token_ids = await tokenize_response_messages(
                runtime,
                prepared.generation_messages,
                reasoning_effort=prepared.params.reasoning_effort,
            )
        except (EngineInputError, InvalidRequestError):
            raise
        except Exception as exc:
            preflight_exception = exc

        return StreamingResponse(
            stream_response_with_persistence(
                prepared=PreparedPersistedStreamingResponse(
                    response_id=response_id,
                    created_at=created_at,
                    request=effective_request,
                    all_tools=all_tools,
                    conversation_messages=conversation_messages,
                    stored_input_items=stored_input_items,
                    prepared_generation=prepared,
                    runtime=runtime,
                    mcp_manager=mcp_manager,
                    preflight_prompt_token_ids=preflight_prompt_token_ids,
                    preflight_exception=preflight_exception,
                    use_agentic_tools=use_agentic_tools,
                    persist_response=persist_response,
                    max_tool_rounds=getattr(mcp_settings, "max_tool_rounds", 0),
                    timeout_seconds=int(getattr(mcp_settings, "timeout_seconds", 30.0)),
                ),
                callbacks=PersistedStreamingCallbacks(
                    tokenize_response_messages=tokenize_response_messages,
                    execute_mcp_tool_call=execute_mcp_tool_call,
                    persist_response_state=lambda response, context_messages, input_items: _store_response_state(
                        store=store,
                        response=response,
                        context_messages=context_messages,
                        input_items=input_items,
                    ),
                    build_failed_response=_build_failed_response,
                    extract_sequence_number=_extract_sequence_number_from_stream_event,
                    extract_response_payload=_extract_response_payload_from_stream_event,
                ),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming mode
    return await create_non_streaming_response_use_case(
        prepared=create_prepared,
        mcp_manager=mcp_manager,
        execute_response_generation=execute_response_generation,
        build_response_from_generation=build_response_from_generation,
        persist_response_state=lambda **kwargs: _store_response_state(store=store, **kwargs),
    )


@router.get("/responses/{response_id}")
async def get_response(
    response_id: str,
    include: list[str] | None = Query(None),
    store: StoreService = Inject("store_service"),
):
    del include
    stored = await store.get(namespace=_STORE_NAMESPACE, key=response_id)
    if stored is None:
        raise NotFoundError(f"Response '{response_id}' not found.")
    return stored


@router.get("/responses/{response_id}/input_items", response_model=ResponseItemList)
async def list_response_input_items(
    response_id: str,
    limit: int = Query(20, ge=1, le=100),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    after: str | None = Query(None),
    before: str | None = Query(None),
    include: list[str] | None = Query(None),
    store: StoreService = Inject("store_service"),
):
    del include
    after = after if isinstance(after, str) else None
    before = before if isinstance(before, str) else None
    return await list_response_input_items_use_case(
        response_id=response_id,
        limit=limit,
        order=order,
        after=after,
        before=before,
        load_stored_response=lambda target_id: store.get(
            namespace=_STORE_NAMESPACE,
            key=target_id,
        ),
        load_response_input_items=lambda target_id: response_store.get_response_input_items(
            store,
            response_id=target_id,
        ),
        paginate_input_items=_paginate_response_input_items,
    )


@router.delete("/responses/{response_id}", response_model=ResponseDeleted)
async def delete_response(
    response_id: str,
    store: StoreService = Inject("store_service"),
    job_store: JobStoreService = Inject("job_store_service"),
    lease_service: JobLeaseService = Inject("job_lease_service"),
):
    return await delete_response_use_case(
        response_id=response_id,
        load_stored_response=lambda target_id: store.get(namespace=_STORE_NAMESPACE, key=target_id),
        load_response_execution=job_store.get_response_execution,
        delete_response_state=lambda target_id: _delete_response_state(
            store,
            job_store,
            lease_service,
            target_id,
        ),
    )


@router.post("/responses/{response_id}/cancel")
async def cancel_response(
    response_id: str,
    store: StoreService = Inject("store_service"),
    job_store: JobStoreService = Inject("job_store_service"),
    runner: JobRunnerService = Inject("job_runner_service"),
):
    return await cancel_response_use_case(
        response_id=response_id,
        request_cancel_execution=job_store.request_cancel_response_execution,
        load_stored_response=lambda target_id: store.get(namespace=_STORE_NAMESPACE, key=target_id),
        persist_response_state=lambda response: _store_response_state(store, response),
        delete_response_spec=job_store.delete_response_spec,
        request_runner_cancel=runner.request_cancel,
        now=time.time,
    )


async def _background_generate(
    spec: ResponseJobSpec,
    engine_router: EngineRouter,
    store: StoreService,
    mcp_manager: MCPClientManager,
    tool_parsing_service: ToolParsingService,
) -> None:
    await background_generate_response_use_case(
        spec=spec,
        store=store,
        mcp_manager=mcp_manager,
        tool_parsing_service=tool_parsing_service,
        resolve_client_for_target=engine_router.client_for_target,
        get_settings=GetSettings,
        should_persist_response=_should_persist_response,
        build_input_items=lambda response_id, messages: _conversation_messages_to_input_items(
            response_id=response_id,
            messages=messages,
        ),
        initialize_engine_client=initialize_engine_client,
        prepare_response_generation=prepare_response_generation,
        execute_response_generation=execute_response_generation,
        build_response_from_generation=build_response_from_generation,
        persist_response_state=lambda *args, **kwargs: _store_response_state(
            *args,
            **(
                {
                    **kwargs,
                    "context_messages": messages_for_response_continuation(kwargs["context_messages"]),
                }
                if kwargs.get("context_messages") is not None
                else kwargs
            ),
        ),
        build_failed_response=_build_failed_response,
        store_namespace=_STORE_NAMESPACE,
    )
