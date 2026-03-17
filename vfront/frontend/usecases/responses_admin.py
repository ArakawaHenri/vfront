"""Administrative use cases for stored Responses API resources."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from vfront.frontend.middleware.exceptions import InvalidRequestError, NotFoundError
from vfront.protocol.response import ResponseDeleted, ResponseItemList, ResponseObject

if TYPE_CHECKING:
    from vfront.frontend.service.jobs.store import ResponseCancelUpdate
    from vfront.protocol.job import ResponseExecution

LoadStoredResponse = Callable[[str], Awaitable[object | None]]
LoadResponseInputItems = Callable[[str], Awaitable[list[dict[str, Any]]]]
PaginateInputItems = Callable[
    ...,
    tuple[list[dict[str, Any]], bool],
]
LoadResponseExecution = Callable[[str], Awaitable["ResponseExecution | None"]]
DeleteResponseState = Callable[[str], Awaitable[bool]]
RequestCancelExecution = Callable[[str], Awaitable["ResponseCancelUpdate | None"]]
PersistResponseState = Callable[[ResponseObject], Awaitable[None]]
DeleteResponseSpec = Callable[[str], Awaitable[bool]]
RequestRunnerCancel = Callable[[str, str], Awaitable[None]]
TimeProvider = Callable[[], float]


async def list_response_input_items_use_case(
    *,
    response_id: str,
    limit: int,
    order: str,
    after: str | None,
    before: str | None,
    load_stored_response: LoadStoredResponse,
    load_response_input_items: LoadResponseInputItems,
    paginate_input_items: PaginateInputItems,
) -> ResponseItemList:
    stored = await load_stored_response(response_id)
    if stored is None:
        raise NotFoundError(f"Response '{response_id}' not found.")

    input_items = await load_response_input_items(response_id)
    page_items, has_more = paginate_input_items(
        items=input_items,
        limit=limit,
        order=order,
        after=after,
        before=before,
    )
    return ResponseItemList(
        data=page_items,
        has_more=has_more,
        first_id=str(page_items[0].get("id")) if page_items else None,
        last_id=str(page_items[-1].get("id")) if page_items else None,
    )


async def delete_response_use_case(
    *,
    response_id: str,
    load_stored_response: LoadStoredResponse,
    load_response_execution: LoadResponseExecution,
    delete_response_state: DeleteResponseState,
) -> ResponseDeleted:
    stored = await load_stored_response(response_id)
    if stored is None:
        raise NotFoundError(f"Response '{response_id}' not found.")

    execution = await load_response_execution(response_id)
    if execution is not None and execution.status in {"queued", "running"}:
        raise InvalidRequestError(
            f"Cannot delete response '{response_id}' while it is {execution.status}. "
            "Cancel it first or wait for completion."
        )

    await delete_response_state(response_id)
    return ResponseDeleted(id=response_id)


async def cancel_response_use_case(
    *,
    response_id: str,
    request_cancel_execution: RequestCancelExecution,
    load_stored_response: LoadStoredResponse,
    persist_response_state: PersistResponseState,
    delete_response_spec: DeleteResponseSpec,
    request_runner_cancel: RequestRunnerCancel,
    now: TimeProvider,
) -> dict[str, Any]:
    cancel_update = await request_cancel_execution(response_id)
    execution = cancel_update.execution if cancel_update is not None else None
    stored = await load_stored_response(response_id)
    current_time = now()

    if execution is None and stored is None:
        raise NotFoundError(f"Response '{response_id}' not found.")
    if execution is None and stored is not None:
        response = ResponseObject.model_validate(stored)
        if response.status not in {"queued", "in_progress"}:
            raise InvalidRequestError(
                f"Response '{response_id}' is not in progress (status: {response.status})."
            )

    if cancel_update is not None and cancel_update.previous_status in {"completed", "failed", "cancelled"}:
        status = execution.status if execution is not None else cancel_update.previous_status
        raise InvalidRequestError(
            f"Response '{response_id}' is not in progress (status: {status})."
        )

    if stored is not None and isinstance(stored, dict):
        response = ResponseObject.model_validate(stored)
        if execution is not None and execution.status == "running":
            cancelling_response = response.model_copy(
                update={"status": "cancelling", "completed_at": None}
            )
            await persist_response_state(cancelling_response)
            await request_runner_cancel("response", response_id)
            return cancelling_response.model_dump()

        cancelled_response = response.model_copy(
            update={"status": "cancelled", "completed_at": current_time}
        )
        await persist_response_state(cancelled_response)
        await delete_response_spec(response_id)
        await request_runner_cancel("response", response_id)
        return cancelled_response.model_dump()

    await request_runner_cancel("response", response_id)
    return {
        "id": response_id,
        "object": "response",
        "status": "cancelling" if execution is not None and execution.status == "running" else "cancelled",
    }
