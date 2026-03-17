"""Responses persistence helpers."""

from __future__ import annotations

from typing import Any

from fastapiex.settings import GetSettings

from vfront.adapter.response import (
    RESPONSE_CONTEXT_NAMESPACE,
    _conversation_messages_to_input_items,
)
from vfront.frontend.middleware.exceptions import InvalidRequestError
from vfront.frontend.service.jobs.lease import JobLeaseService
from vfront.frontend.service.jobs.store import JobStoreService
from vfront.frontend.service.store.lmdb import StoreService
from vfront.protocol.response import ResponseCreateRequest, ResponseObject

RESPONSE_STORE_NAMESPACE = "responses"
RESPONSE_INPUT_ITEMS_NAMESPACE = "responses_input_items"


def should_persist_response(request: ResponseCreateRequest) -> bool:
    """Whether this response should be persisted."""
    return request.store is not False


def paginate_response_input_items(
    *,
    items: list[dict[str, Any]],
    limit: int,
    order: str,
    after: str | None,
    before: str | None,
) -> tuple[list[dict[str, Any]], bool]:
    if after is not None and before is not None:
        raise InvalidRequestError(
            "Cannot specify both 'after' and 'before' when listing response input items.",
            param="after",
        )

    ordered = list(items) if order == "asc" else list(reversed(items))
    ids = [str(item.get("id") or "") for item in ordered]

    if after is not None:
        try:
            ordered = ordered[ids.index(after) + 1 :]
        except ValueError as exc:
            raise InvalidRequestError(
                "Invalid 'after' cursor for response input items.",
                param="after",
            ) from exc
        ids = [str(item.get("id") or "") for item in ordered]

    if before is not None:
        try:
            ordered = ordered[: ids.index(before)]
        except ValueError as exc:
            raise InvalidRequestError(
                "Invalid 'before' cursor for response input items.",
                param="before",
            ) from exc

    has_more = len(ordered) > limit
    return ordered[:limit], has_more


async def store_response_state(
    store: StoreService,
    response: ResponseObject,
    context_messages: list[dict[str, Any]] | None = None,
    input_items: list[dict[str, Any]] | None = None,
) -> None:
    retention = GetSettings("frontend.store").default_retention_minutes
    await store.set(
        namespace=RESPONSE_STORE_NAMESPACE,
        key=response.id,
        value=response.model_dump(),
        retention=retention,
    )
    if context_messages is not None:
        await store.set(
            namespace=RESPONSE_CONTEXT_NAMESPACE,
            key=response.id,
            value=context_messages,
            retention=retention,
        )
    if input_items is not None:
        await store.set(
            namespace=RESPONSE_INPUT_ITEMS_NAMESPACE,
            key=response.id,
            value=input_items,
            retention=retention,
        )


async def delete_response_state(
    store: StoreService,
    job_store: JobStoreService,
    lease_service: JobLeaseService,
    response_id: str,
) -> bool:
    response_deleted = await store.delete(namespace=RESPONSE_STORE_NAMESPACE, key=response_id)
    context_deleted = await store.delete(
        namespace=RESPONSE_CONTEXT_NAMESPACE,
        key=response_id,
    )
    input_items_deleted = await store.delete(
        namespace=RESPONSE_INPUT_ITEMS_NAMESPACE,
        key=response_id,
    )
    job_deleted = await job_store.delete_response_spec(response_id)
    exec_deleted = await job_store.delete_response_execution(response_id)
    lease_deleted = await lease_service.clear_response(response_id)
    return bool(
        response_deleted
        or context_deleted
        or input_items_deleted
        or job_deleted
        or exec_deleted
        or lease_deleted
    )


async def get_response_input_items(
    store: StoreService,
    *,
    response_id: str,
) -> list[dict[str, Any]]:
    input_items = await store.get(namespace=RESPONSE_INPUT_ITEMS_NAMESPACE, key=response_id)
    if isinstance(input_items, list):
        return input_items

    stored_context = await store.get(
        namespace=RESPONSE_CONTEXT_NAMESPACE,
        key=response_id,
    )
    return (
        _conversation_messages_to_input_items(
            response_id=response_id,
            messages=stored_context,
        )
        if isinstance(stored_context, list)
        else []
    )
