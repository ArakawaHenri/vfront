"""POST /v1/chat/completions — Chat Completions endpoint.

No separate handler layer. Router directly calls adapter + engine.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request
from fastapiex.di import Inject
from fastapiex.settings import GetSettings

from vfront.adapter.output import request_output_to_chat_response
from vfront.adapter.sampling import resolve_deprecated_functions
from vfront.frontend.api.v1.helper.message_paging import (
    decode_message_cursor,
    page_message_items,
)
from vfront.frontend.compat.validators import validate_chat_request
from vfront.frontend.service.engine.router import EngineRouter
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.mcp.executor import execute_mcp_tool_call
from vfront.frontend.service.persistence import chat_store
from vfront.frontend.service.store.lmdb import StoreService
from vfront.frontend.service.tool_parsing.service import ToolParsingService
from vfront.frontend.usecases import (
    create_chat_completion_use_case,
    prepare_chat_completion_request,
)
from vfront.protocol.chat_completion import (
    ChatCompletionDeleted,
    ChatCompletionListResponse,
    ChatCompletionMessageListResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    StoredChatCompletionObject,
    UpdateStoredChatCompletionRequest,
)
from vfront.shared.engine.generation import collect_final_stream_output
from vfront.shared.engine.tokenization import tokenize_chat_messages

logger = logging.getLogger(__name__)

router = APIRouter()

_STORE_NAMESPACE = chat_store.CHAT_COMPLETION_STORE_NAMESPACE
_INDEX_NAMESPACE = chat_store.CHAT_COMPLETION_INDEX_NAMESPACE
_MESSAGES_NAMESPACE = chat_store.CHAT_COMPLETION_MESSAGES_NAMESPACE
_INDEX_SCALE = chat_store.CHAT_COMPLETION_INDEX_SCALE

async def _store_completion(
    store: StoreService,
    request: ChatCompletionRequest,
    response: ChatCompletionResponse,
    messages: list[dict],
) -> None:
    await chat_store.store_completion(store, request, response, messages)


async def _ensure_chat_index(store: StoreService) -> None:
    await chat_store.ensure_chat_index(store)


async def _update_stored_completion_metadata(
    store: StoreService,
    *,
    completion_id: str,
    metadata: dict[str, str],
) -> StoredChatCompletionObject:
    return await chat_store.update_stored_completion_metadata(
        store,
        completion_id=completion_id,
        metadata=metadata,
    )


async def _delete_chat_completion_state(
    store: StoreService,
    completion_id: str,
) -> bool:
    return await chat_store.delete_chat_completion_state(store, completion_id)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    engine_router: EngineRouter = Inject("engine_router"),
    store: StoreService = Inject("store_service"),
    mcp_manager: MCPClientManager = Inject("mcp_client_manager"),
    tool_parsing_service: ToolParsingService = Inject("tool_parsing_service"),
):
    # Map deprecated functions/function_call → tools/tool_choice
    resolve_deprecated_functions(request)
    validate_chat_request(request)
    prepared = await prepare_chat_completion_request(
        request=request,
        engine_router=engine_router,
        mcp_manager=mcp_manager,
        tool_parsing_service=tool_parsing_service,
        get_settings=GetSettings,
        tokenize_chat_messages=tokenize_chat_messages,
    )
    return await create_chat_completion_use_case(
        request=request,
        prepared=prepared,
        store=store,
        execute_mcp_tool_call_fn=execute_mcp_tool_call,
        request_output_to_chat_response_fn=request_output_to_chat_response,
        collect_final_stream_output_fn=collect_final_stream_output,
        store_completion_fn=_store_completion,
    )


def _build_list_predicate(
    model_filter: str | None,
    metadata_filter: dict[str, str],
) -> Callable[[StoredChatCompletionObject], bool] | None:
    """Build a predicate for list_created_at_models, or return None if no filter."""
    if not model_filter and not metadata_filter:
        return None

    def predicate(obj: StoredChatCompletionObject) -> bool:
        if model_filter and obj.model != model_filter:
            return False
        if metadata_filter:
            obj_meta = obj.metadata or {}
            if not all(obj_meta.get(k) == v for k, v in metadata_filter.items()):
                return False
        return True

    return predicate


@router.get("/chat/completions", response_model=ChatCompletionListResponse)
async def list_chat_completions(
    raw_request: Request,
    after: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    order: Literal["asc", "desc"] = Query("asc"),
    model: str | None = Query(None),
    metadata: Annotated[
        str | None,
        Query(
            description=(
                "A list of metadata keys to filter the Chat Completions by. "
                "Example: `metadata[key1]=value1&metadata[key2]=value2`"
            ),
            json_schema_extra={
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        ),
    ] = None,
    store: StoreService = Inject("store_service"),
):
    del metadata
    # Parse metadata[key]=value filters from query string
    metadata_filter: dict[str, str] = {
        key[9:-1]: value
        for key, value in raw_request.query_params.items()
        if key.startswith("metadata[") and key.endswith("]")
    }
    predicate = _build_list_predicate(model, metadata_filter)

    completions, has_more = await chat_store.list_chat_completions(
        store,
        limit=limit,
        after=after,
        order=order,
        predicate=predicate,
    )
    return ChatCompletionListResponse(
        data=completions,
        has_more=has_more,
        first_id=completions[0].id if completions else None,
        last_id=completions[-1].id if completions else None,
    )


@router.get("/chat/completions/{completion_id}", response_model=StoredChatCompletionObject)
async def get_chat_completion(
    completion_id: str,
    store: StoreService = Inject("store_service"),
):
    return await chat_store.get_chat_completion(store, completion_id)


@router.post("/chat/completions/{completion_id}", response_model=StoredChatCompletionObject)
async def update_chat_completion(
    completion_id: str,
    request: UpdateStoredChatCompletionRequest,
    store: StoreService = Inject("store_service"),
):
    return await _update_stored_completion_metadata(
        store,
        completion_id=completion_id,
        metadata=request.metadata,
    )


@router.delete("/chat/completions/{completion_id}", response_model=ChatCompletionDeleted)
async def delete_chat_completion(
    completion_id: str,
    store: StoreService = Inject("store_service"),
):
    deleted = await _delete_chat_completion_state(store, completion_id)
    return ChatCompletionDeleted(id=completion_id, deleted=deleted)


@router.get(
    "/chat/completions/{completion_id}/messages",
    response_model=ChatCompletionMessageListResponse,
)
async def list_chat_completion_messages(
    completion_id: str,
    limit: int = Query(20, ge=1, le=100),
    after: str | None = Query(None),
    store: StoreService = Inject("store_service"),
):
    # Verify the completion exists
    all_data = await chat_store.get_chat_completion_messages(store, completion_id)

    start_index = 0
    if after is not None:
        start_index = decode_message_cursor(completion_id, after) + 1

    end_index = start_index + limit
    page_data = page_message_items(
        all_data[start_index:end_index],
        completion_id=completion_id,
        start_index=start_index,
    )
    has_more = len(all_data) > end_index

    first_id = page_data[0]["id"] if page_data else None
    last_id = page_data[-1]["id"] if page_data else None

    return ChatCompletionMessageListResponse(
        data=page_data,
        has_more=has_more,
        first_id=first_id,
        last_id=last_id,
    )
