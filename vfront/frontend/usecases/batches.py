"""Batch orchestration shared by batch route handlers."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from vfront.adapter.batch import (
    build_error_jsonl,
    build_output_jsonl,
    parse_batch_input,
    validate_batch_endpoint,
)
from vfront.frontend.middleware.exceptions import InvalidRequestError, NotFoundError
from vfront.frontend.service.engine.router import EngineRouter
from vfront.frontend.service.file_storage.main import FileStorageService
from vfront.frontend.service.jobs.runner import JobRunnerService
from vfront.frontend.service.jobs.store import JobStoreService
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.store.lmdb import StoreService
from vfront.frontend.service.tool_parsing.service import ToolParsingService
from vfront.protocol.batch import Batch, BatchError, BatchErrors, BatchRequestCounts, BatchUsage
from vfront.protocol.job import BatchExecution, BatchJobSpec, JobResolvedTarget
from vfront.shared.engine.types import EngineHealth

logger = logging.getLogger(__name__)


def build_batch_job_spec(
    batch_id: str,
    endpoint: str,
    input_file_id: str,
    completion_window: str,
    metadata: dict[str, str] | None,
    request_targets: dict[str, JobResolvedTarget],
) -> BatchJobSpec:
    return BatchJobSpec(
        batch_id=batch_id,
        endpoint=endpoint,
        input_file_id=input_file_id,
        completion_window=completion_window,
        metadata=metadata,
        request_targets=request_targets,
    )


def build_batch_request_targets(
    batch_requests: list[Any],
    engine_router: EngineRouter,
) -> dict[str, JobResolvedTarget]:
    targets: dict[str, JobResolvedTarget] = {}
    for request in batch_requests:
        model = request.body.get("model")
        if not isinstance(model, str) or not model:
            continue
        try:
            target = engine_router.resolve_target(model)
        except Exception:
            continue
        targets[request.custom_id] = JobResolvedTarget(
            requested_model=target.requested_model,
            public_name=target.public_name,
            runtime_model=target.runtime_model,
            backend=target.backend,
            adapter=target.adapter,
            tool_parser=target.tool_parser,
        )
    return targets


async def create_batch_use_case(
    *,
    request: dict[str, Any],
    engine_router: EngineRouter,
    store: StoreService,
    file_storage: FileStorageService,
    job_store: JobStoreService,
    runner: JobRunnerService,
    store_batch_state: Callable[[StoreService, Batch], Awaitable[None]],
) -> Batch:
    input_file_id = request.get("input_file_id")
    endpoint = request.get("endpoint")
    completion_window = request.get("completion_window", "24h")
    metadata = request.get("metadata")

    if not input_file_id:
        raise InvalidRequestError("'input_file_id' is required.")
    if not endpoint:
        raise InvalidRequestError("'endpoint' is required.")

    try:
        validate_batch_endpoint(endpoint)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    file_meta = await file_storage.get_metadata(input_file_id)
    if file_meta is None:
        raise NotFoundError(f"File '{input_file_id}' not found.")

    content = await file_storage.get_content(input_file_id)
    if content is None:
        raise NotFoundError(f"Content for file '{input_file_id}' not found.")

    try:
        batch_requests = parse_batch_input(content)
    except ValueError as exc:
        raise InvalidRequestError(f"Invalid batch input file: {exc}") from exc

    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    now = int(time.time())
    batch = Batch(
        id=batch_id,
        endpoint=endpoint,
        input_file_id=input_file_id,
        completion_window=completion_window,
        status="validating",
        created_at=now,
        expires_at=now + 86400,
        request_counts=BatchRequestCounts(total=len(batch_requests)),
        metadata=metadata,
    )
    request_targets = build_batch_request_targets(batch_requests, engine_router)
    job_spec = build_batch_job_spec(
        batch_id=batch_id,
        endpoint=endpoint,
        input_file_id=input_file_id,
        completion_window=completion_window,
        metadata=metadata,
        request_targets=request_targets,
    )
    execution = BatchExecution(
        batch_id=batch_id,
        status="queued",
        endpoint=endpoint,
        created_at=float(now),
    )

    await store_batch_state(store, batch)
    await job_store.put_batch_spec(job_spec)
    await job_store.put_batch_execution(execution)
    await runner.notify()
    return batch


async def process_batch_use_case(
    *,
    spec: BatchJobSpec,
    store: StoreService,
    file_storage: FileStorageService,
    engine_router: EngineRouter,
    mcp_manager: MCPClientManager,
    tool_parsing_service: ToolParsingService,
    batch_namespace: str,
    store_batch_state: Callable[[StoreService, Batch], Awaitable[None]],
    dispatch_request_fn: Callable[[str, dict[str, Any], JobResolvedTarget | None, dict[str, EngineHealth]], Awaitable[dict[str, Any]]],
    get_settings: Callable[[str], Any],
    process_batch_requests_fn: Callable[..., Awaitable[list[Any]]],
    build_output_jsonl_fn: Callable[[list[Any]], bytes] = build_output_jsonl,
    build_error_jsonl_fn: Callable[[list[Any]], bytes | None] = build_error_jsonl,
) -> None:
    batch_id = spec.batch_id
    del engine_router, mcp_manager, tool_parsing_service
    try:
        content = await file_storage.get_content(spec.input_file_id)
        if content is None:
            raise NotFoundError(f"Content for file '{spec.input_file_id}' not found.")
        try:
            batch_requests = parse_batch_input(content)
        except ValueError as exc:
            raise InvalidRequestError(f"Invalid batch input file: {exc}") from exc

        data = await store.get(namespace=batch_namespace, key=batch_id)
        if data is None:
            return
        batch = Batch.model_validate(data)
        batch.status = "in_progress"
        batch.in_progress_at = int(time.time())
        await store_batch_state(store, batch)

        target_by_custom_id = spec.request_targets
        custom_id_by_body_id = {id(request.body): request.custom_id for request in batch_requests}
        health_by_backend: dict[str, EngineHealth] = {}

        async def dispatch(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
            custom_id = custom_id_by_body_id.get(id(body))
            request_target = (
                target_by_custom_id.get(custom_id) if custom_id is not None else None
            )
            return await dispatch_request_fn(
                endpoint,
                body,
                request_target,
                health_by_backend,
            )

        outputs = await process_batch_requests_fn(
            requests=batch_requests,
            endpoint=spec.endpoint,
            dispatch_fn=dispatch,
            max_concurrency=get_settings("frontend.routing").batch_concurrency,
        )

        batch.status = "finalizing"
        batch.finalizing_at = int(time.time())
        completed = sum(1 for output in outputs if output.error is None)
        failed = sum(1 for output in outputs if output.error is not None)
        batch.request_counts = BatchRequestCounts(
            total=len(outputs),
            completed=completed,
            failed=failed,
        )

        total_input = 0
        total_output = 0
        for output in outputs:
            if output.response and output.response.body:
                usage = output.response.body.get("usage", {})
                total_input += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                total_output += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        if total_input or total_output:
            batch.usage = BatchUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            )
        await store_batch_state(store, batch)

        output_jsonl = build_output_jsonl_fn(outputs)
        output_file = await file_storage.upload(
            filename=f"{batch_id}_output.jsonl",
            content=output_jsonl,
            purpose="batch_output",
        )
        batch.output_file_id = output_file.id

        error_jsonl = build_error_jsonl_fn(outputs)
        if error_jsonl:
            error_file = await file_storage.upload(
                filename=f"{batch_id}_errors.jsonl",
                content=error_jsonl,
                purpose="batch_output",
            )
            batch.error_file_id = error_file.id

        batch.status = "completed"
        batch.completed_at = int(time.time())
        await store_batch_state(store, batch)
        logger.info(
            "Batch completed: id=%s total=%d ok=%d fail=%d",
            batch_id,
            len(outputs),
            completed,
            failed,
        )

    except asyncio.CancelledError:
        logger.info("Batch cancelled: %s", batch_id)
        raise
    except Exception:
        logger.exception("Batch failed: %s", batch_id)
        data = await store.get(namespace=batch_namespace, key=batch_id)
        if data:
            batch = Batch.model_validate(data)
            batch.status = "failed"
            batch.failed_at = int(time.time())
            batch.errors = BatchErrors(
                data=[BatchError(code="server_error", message="Internal batch processing error")]
            )
            await store_batch_state(store, batch)
        raise
