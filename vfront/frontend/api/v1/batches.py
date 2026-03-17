"""POST/GET /v1/batches — Batch API endpoints.

Implements:
- POST /v1/batches — create batch job
- GET /v1/batches/{batch_id} — retrieve batch status
- GET /v1/batches — list batches
- POST /v1/batches/{batch_id}/cancel — cancel batch
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Query
from fastapiex.di import Inject
from fastapiex.settings import GetSettings

from vfront.adapter.batch import (
    build_error_jsonl,
    build_output_jsonl,
    process_batch_requests,
)
from vfront.adapter.output import (
    embedding_outputs_to_response,
    request_output_to_chat_response,
)
from vfront.adapter.pooling import (
    embedding_request_to_encode_params,
    normalize_embedding_inputs,
    resolve_embedding_encoding_format,
)
from vfront.adapter.sampling import resolve_deprecated_functions
from vfront.frontend.api.v1.helper import (
    build_response_from_generation,
    ensure_created_at_index,
    execute_response_generation,
    list_created_at_models,
    store_created_at_indexed_model,
)
from vfront.frontend.compat.validators import (
    validate_chat_request,
    validate_responses_request,
)
from vfront.frontend.middleware.exceptions import InvalidRequestError, NotFoundError
from vfront.frontend.service.engine.client_initializer import initialize_engine_client
from vfront.frontend.service.engine.router import EngineRouter, ResolvedEngineClient
from vfront.frontend.service.file_storage.main import FileStorageService
from vfront.frontend.service.jobs.runner import JobRunnerService
from vfront.frontend.service.jobs.store import JobStoreService
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.mcp.executor import execute_mcp_tool_call
from vfront.frontend.service.store.lmdb import StoreService
from vfront.frontend.service.tool_parsing.service import ToolParsingService
from vfront.frontend.usecases import (
    PreparedResponseCreation,
    build_batch_job_spec,
    build_batch_request_targets,
    create_batch_use_case,
    create_chat_completion_use_case,
    create_non_streaming_response_use_case,
    prepare_chat_completion_request,
    prepare_responses_request,
    process_batch_use_case,
)
from vfront.frontend.usecases.completions import (
    execute_non_streaming_completion,
    prepare_completion_use_case,
)
from vfront.protocol.batch import (
    Batch,
    BatchListResponse,
)
from vfront.protocol.chat_completion import ChatCompletionRequest
from vfront.protocol.completion import CompletionRequest
from vfront.protocol.embedding import EmbeddingRequest
from vfront.protocol.job import BatchJobSpec, JobResolvedTarget
from vfront.protocol.response import (
    ResponseCreateRequest,
    ResponseObject,
)
from vfront.shared.engine.generation import collect_final_stream_output
from vfront.shared.engine.tokenization import tokenize_chat_messages
from vfront.shared.engine.types import EngineHealth

logger = logging.getLogger(__name__)

router = APIRouter()

_BATCH_NAMESPACE = "batches"
_BATCH_INDEX_NAMESPACE = "batches_idx_created_at"
_BATCH_INDEX_SCALE = 1


async def _store_batch_state(
    store: StoreService,
    batch: Batch,
) -> None:
    await store_created_at_indexed_model(
        store,
        store_namespace=_BATCH_NAMESPACE,
        index_namespace=_BATCH_INDEX_NAMESPACE,
        model=batch,
        retention=GetSettings("frontend.store").default_retention_minutes,
        object_id_getter=lambda item: item.id,
        created_at_getter=lambda item: item.created_at,
        payload_getter=lambda item: item.model_dump(),
        index_scale=_BATCH_INDEX_SCALE,
    )


async def _ensure_batch_index(store: StoreService) -> None:
    await ensure_created_at_index(
        store,
        store_namespace=_BATCH_NAMESPACE,
        index_namespace=_BATCH_INDEX_NAMESPACE,
        retention=GetSettings("frontend.store").default_retention_minutes,
        model_validate=Batch.model_validate,
        object_id_getter=lambda item: item.id,
        created_at_getter=lambda item: item.created_at,
        index_scale=_BATCH_INDEX_SCALE,
    )


async def _noop_store_completion(*args, **kwargs) -> None:
    del args, kwargs


async def _noop_persist_response_state(*args, **kwargs) -> None:
    del args, kwargs


def _assert_batch_previous_response_reusable(
    response_id: str,
    response: ResponseObject,
) -> None:
    if response.status not in {"completed", "incomplete"}:
        raise InvalidRequestError(
            f"Response '{response_id}' is not ready for continuation (status: {response.status})."
        )


def _build_batch_job_spec(
    batch_id: str,
    endpoint: str,
    input_file_id: str,
    completion_window: str,
    metadata: dict[str, str] | None,
    request_targets: dict[str, JobResolvedTarget],
) -> BatchJobSpec:
    return build_batch_job_spec(
        batch_id=batch_id,
        endpoint=endpoint,
        input_file_id=input_file_id,
        completion_window=completion_window,
        metadata=metadata,
        request_targets=request_targets,
    )


def _build_batch_request_targets(
    batch_requests,
    engine_router: EngineRouter,
) -> dict[str, JobResolvedTarget]:
    return build_batch_request_targets(batch_requests, engine_router)


@router.post("/batches", response_model=Batch)
async def create_batch(
    request: dict,
    engine_router: EngineRouter = Inject("engine_router"),
    store: StoreService = Inject("store_service"),
    file_storage: FileStorageService = Inject("file_storage_service"),
    job_store: JobStoreService = Inject("job_store_service"),
    runner: JobRunnerService = Inject("job_runner_service"),
):
    return await create_batch_use_case(
        request=request,
        engine_router=engine_router,
        store=store,
        file_storage=file_storage,
        job_store=job_store,
        runner=runner,
        store_batch_state=_store_batch_state,
    )


@router.get("/batches/{batch_id}")
async def get_batch(
    batch_id: str,
    store: StoreService = Inject("store_service"),
):
    data = await store.get(namespace=_BATCH_NAMESPACE, key=batch_id)
    if data is None:
        raise NotFoundError(f"Batch '{batch_id}' not found.")
    return data


@router.get("/batches", response_model=BatchListResponse)
async def list_batches(
    after: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    store: StoreService = Inject("store_service"),
):
    await _ensure_batch_index(store)
    batches, has_more = await list_created_at_models(
        store,
        store_namespace=_BATCH_NAMESPACE,
        index_namespace=_BATCH_INDEX_NAMESPACE,
        limit=limit,
        after=after,
        model_validate=Batch.model_validate,
        object_id_getter=lambda item: item.id,
        created_at_getter=lambda item: item.created_at,
        index_scale=_BATCH_INDEX_SCALE,
    )

    return BatchListResponse(
        data=batches,
        has_more=has_more,
        first_id=batches[0].id if batches else None,
        last_id=batches[-1].id if batches else None,
    )


@router.post("/batches/{batch_id}/cancel", response_model=Batch)
async def cancel_batch(
    batch_id: str,
    store: StoreService = Inject("store_service"),
    job_store: JobStoreService = Inject("job_store_service"),
    runner: JobRunnerService = Inject("job_runner_service"),
):
    data = await store.get(namespace=_BATCH_NAMESPACE, key=batch_id)
    if data is None:
        raise NotFoundError(f"Batch '{batch_id}' not found.")

    batch = Batch.model_validate(data)
    if batch.status not in ("validating", "in_progress"):
        raise InvalidRequestError(f"Cannot cancel batch with status '{batch.status}'.")

    cancel_update = await job_store.request_cancel_batch_execution(batch_id)
    execution = cancel_update.execution if cancel_update is not None else None
    now = int(time.time())
    if cancel_update is not None and cancel_update.previous_status in {"completed", "failed", "cancelled"}:
        raise InvalidRequestError(f"Cannot cancel batch with status '{execution.status}'.")

    if execution is not None and execution.status == "running":
        batch.status = "cancelling"
        batch.cancelling_at = now
    else:
        batch.status = "cancelled"
        batch.cancelled_at = now
        await job_store.delete_batch_spec(batch_id)
    await _store_batch_state(store, batch)
    await runner.request_cancel("batch", batch_id)

    return batch


async def _process_batch(
    spec: BatchJobSpec,
    engine_router: EngineRouter,
    store: StoreService,
    file_storage: FileStorageService,
    mcp_manager: MCPClientManager,
    tool_parsing_service: ToolParsingService,
) -> None:
    """Background batch processing task."""
    await process_batch_use_case(
        spec=spec,
        store=store,
        file_storage=file_storage,
        engine_router=engine_router,
        mcp_manager=mcp_manager,
        tool_parsing_service=tool_parsing_service,
        batch_namespace=_BATCH_NAMESPACE,
        store_batch_state=_store_batch_state,
        dispatch_request_fn=lambda endpoint, body, request_target, health_by_backend: _dispatch_request(
            endpoint,
            body,
            engine_router,
            store,
            mcp_manager,
            tool_parsing_service=tool_parsing_service,
            request_target=request_target,
            health_by_backend=health_by_backend,
        ),
        get_settings=GetSettings,
        process_batch_requests_fn=process_batch_requests,
        build_output_jsonl_fn=build_output_jsonl,
        build_error_jsonl_fn=build_error_jsonl,
    )


async def _dispatch_request(
    endpoint: str,
    body: dict,
    engine_router: EngineRouter,
    store: StoreService,
    mcp_manager: MCPClientManager | None = None,
    tool_parsing_service: ToolParsingService | None = None,
    request_target: JobResolvedTarget | None = None,
    health_by_backend: dict[str, EngineHealth] | None = None,
) -> dict:
    """Dispatch a single batch request to the appropriate handler."""
    if endpoint == "/v1/chat/completions":
        return await _dispatch_chat_completion_request(
            body=body,
            engine_router=engine_router,
            mcp_manager=mcp_manager,
            tool_parsing_service=tool_parsing_service,
            request_target=request_target,
            health_by_backend=health_by_backend,
        )
    if endpoint == "/v1/embeddings":
        return await _dispatch_embedding_request(
            body=body,
            engine_router=engine_router,
            request_target=request_target,
        )
    if endpoint == "/v1/completions":
        return await _dispatch_completion_request(
            body=body,
            engine_router=engine_router,
            tool_parsing_service=tool_parsing_service,
            request_target=request_target,
            health_by_backend=health_by_backend,
        )
    if endpoint == "/v1/responses":
        return await _dispatch_response_request(
            body=body,
            engine_router=engine_router,
            store=store,
            mcp_manager=mcp_manager,
            tool_parsing_service=tool_parsing_service,
            request_target=request_target,
            health_by_backend=health_by_backend,
        )
    raise InvalidRequestError(f"Unknown batch endpoint: {endpoint}")


def _resolve_batch_client(
    *,
    model: str,
    engine_router: EngineRouter,
    request_target: JobResolvedTarget | None,
) -> ResolvedEngineClient:
    if request_target is not None:
        return engine_router.client_for_target(request_target)
    return engine_router.resolve_client(model)


async def _initialize_batch_client(
    *,
    model: str,
    engine_router: EngineRouter,
    tool_parsing_service: ToolParsingService | None,
    request_target: JobResolvedTarget | None,
    require_tools: bool = False,
    health_by_backend: dict[str, EngineHealth] | None = None,
):
    return await initialize_engine_client(
        resolved=_resolve_batch_client(
            model=model,
            engine_router=engine_router,
            request_target=request_target,
        ),
        tool_parsing_service=tool_parsing_service,
        require_tools=require_tools,
        health_by_backend=health_by_backend,
    )


async def _dispatch_chat_completion_request(
    *,
    body: dict,
    engine_router: EngineRouter,
    mcp_manager: MCPClientManager | None,
    tool_parsing_service: ToolParsingService | None,
    request_target: JobResolvedTarget | None,
    health_by_backend: dict[str, EngineHealth] | None,
) -> dict:
    req = ChatCompletionRequest.model_validate(body)
    req.stream = False
    req.store = False
    resolve_deprecated_functions(req)
    validate_chat_request(req)
    prepared = await prepare_chat_completion_request(
        request=req,
        engine_router=engine_router,
        mcp_manager=mcp_manager,
        tool_parsing_service=tool_parsing_service,
        get_settings=GetSettings,
        tokenize_chat_messages=tokenize_chat_messages,
        resolve_engine_client_fn=lambda **kwargs: _initialize_batch_client(
            model=kwargs["model"],
            engine_router=engine_router,
            tool_parsing_service=tool_parsing_service,
            request_target=request_target,
            require_tools=kwargs["require_tools"],
            health_by_backend=health_by_backend,
        ),
    )
    response = await create_chat_completion_use_case(
        request=req,
        prepared=prepared,
        store=None,
        execute_mcp_tool_call_fn=execute_mcp_tool_call,
        request_output_to_chat_response_fn=request_output_to_chat_response,
        collect_final_stream_output_fn=collect_final_stream_output,
        store_completion_fn=_noop_store_completion,
    )
    return response.model_dump()


async def _dispatch_embedding_request(
    *,
    body: dict,
    engine_router: EngineRouter,
    request_target: JobResolvedTarget | None,
) -> dict:
    req = EmbeddingRequest.model_validate(body)
    runtime = _resolve_batch_client(
        model=req.model,
        engine_router=engine_router,
        request_target=request_target,
    ).client
    inputs = normalize_embedding_inputs(req.input)
    encode_params = embedding_request_to_encode_params(req)
    outputs = []
    for index, item in enumerate(inputs):
        if isinstance(item, str):
            token_ids = await runtime.tokenize_text(item)
        else:
            token_ids = item
        result = await runtime.encode(token_ids, encode_params, f"embd-{index}")
        outputs.append(result)
    return embedding_outputs_to_response(
        outputs,
        model=req.model,
        encoding_format=resolve_embedding_encoding_format(req),
    ).model_dump()


async def _dispatch_completion_request(
    *,
    body: dict,
    engine_router: EngineRouter,
    tool_parsing_service: ToolParsingService | None,
    request_target: JobResolvedTarget | None,
    health_by_backend: dict[str, EngineHealth] | None,
) -> dict:
    req = CompletionRequest.model_validate(body)
    req.stream = False
    initialized = await _initialize_batch_client(
        model=req.model,
        engine_router=engine_router,
        tool_parsing_service=tool_parsing_service,
        request_target=request_target,
        health_by_backend=health_by_backend,
    )
    resolved = initialized.resolved
    prepared = await prepare_completion_use_case(
        request=req,
        runtime=initialized.client,
        adapter=resolved.adapter,
        system_fingerprint=initialized.system_fingerprint,
    )
    return (
        await execute_non_streaming_completion(
            request=req,
            prepared=prepared,
        )
    ).model_dump()


async def _dispatch_response_request(
    *,
    body: dict,
    engine_router: EngineRouter,
    store: StoreService,
    mcp_manager: MCPClientManager | None,
    tool_parsing_service: ToolParsingService | None,
    request_target: JobResolvedTarget | None,
    health_by_backend: dict[str, EngineHealth] | None,
) -> dict:
    req = ResponseCreateRequest.model_validate(body)
    req.stream = False
    req.background = False
    req.store = False
    validate_responses_request(req)
    prepared_request = await prepare_responses_request(
        request=req,
        engine_router=engine_router,
        store=store,
        mcp_manager=mcp_manager,
        tool_parsing_service=tool_parsing_service,
        store_namespace="responses",
        get_settings=GetSettings,
        validate_previous_response=_assert_batch_previous_response_reusable,
        build_input_items=lambda _response_id, _messages: [],
        should_persist_response=lambda _request: False,
        resolve_engine_client_fn=lambda **kwargs: _initialize_batch_client(
            model=kwargs["model"],
            engine_router=engine_router,
            tool_parsing_service=tool_parsing_service,
            request_target=request_target,
            require_tools=kwargs["require_tools"],
            health_by_backend=health_by_backend,
        ),
    )
    mcp_settings = GetSettings("frontend.mcp")
    create_prepared = PreparedResponseCreation(
        request=req,
        effective_request=prepared_request.effective_request,
        all_tools=prepared_request.all_tools,
        conversation_messages=prepared_request.conversation_messages,
        stored_input_items=prepared_request.stored_input_items,
        prepared_generation=prepared_request.prepared_generation,
        resolved_target=prepared_request.initialized_client.resolved.target,
        runtime=prepared_request.initialized_client.client,
        response_id=prepared_request.response_id,
        created_at=prepared_request.created_at,
        use_agentic_tools=prepared_request.use_agentic_tools,
        persist_response=prepared_request.persist_response,
        mcp_settings=mcp_settings,
    )
    response = await create_non_streaming_response_use_case(
        prepared=create_prepared,
        mcp_manager=mcp_manager,
        execute_response_generation=execute_response_generation,
        build_response_from_generation=build_response_from_generation,
        persist_response_state=_noop_persist_response_state,
    )
    return response.model_dump()
