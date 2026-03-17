"""Chat completion persistence helpers."""

from __future__ import annotations

from collections.abc import Callable

from fastapiex.settings import GetSettings

from vfront.frontend.api.v1.helper.indexed_store import (
    build_created_at_index_key,
    ensure_created_at_index,
    list_created_at_models,
    store_created_at_indexed_model,
    upsert_created_at_index,
)
from vfront.frontend.middleware.exceptions import NotFoundError
from vfront.frontend.service.store.lmdb import StoreService
from vfront.protocol.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    StoredChatCompletionObject,
)

CHAT_COMPLETION_STORE_NAMESPACE = "chat_completions"
CHAT_COMPLETION_INDEX_NAMESPACE = "chat_completions_idx"
CHAT_COMPLETION_MESSAGES_NAMESPACE = "chat_completions_messages"
CHAT_COMPLETION_INDEX_SCALE = 1


async def store_completion(
    store: StoreService,
    request: ChatCompletionRequest,
    response: ChatCompletionResponse,
    messages: list[dict],
) -> None:
    stored_obj = StoredChatCompletionObject.model_validate({
        **response.model_dump(),
        "metadata": request.metadata,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed,
        "tools": [t.model_dump() for t in request.tools] if request.tools else None,
        "tool_choice": request.tool_choice,
        "response_format": request.response_format.model_dump() if request.response_format else None,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "input_user": request.user,
        "reasoning_effort": request.reasoning_effort,
    })
    retention = GetSettings("frontend.store").default_retention_minutes
    await store_created_at_indexed_model(
        store,
        store_namespace=CHAT_COMPLETION_STORE_NAMESPACE,
        index_namespace=CHAT_COMPLETION_INDEX_NAMESPACE,
        model=stored_obj,
        retention=retention,
        object_id_getter=lambda r: r.id,
        created_at_getter=lambda r: float(r.created),
        payload_getter=lambda r: r.model_dump(),
        index_scale=CHAT_COMPLETION_INDEX_SCALE,
    )
    await store.set(
        namespace=CHAT_COMPLETION_MESSAGES_NAMESPACE,
        key=response.id,
        value=messages,
        retention=retention,
    )


async def ensure_chat_index(store: StoreService) -> None:
    await ensure_created_at_index(
        store,
        store_namespace=CHAT_COMPLETION_STORE_NAMESPACE,
        index_namespace=CHAT_COMPLETION_INDEX_NAMESPACE,
        retention=GetSettings("frontend.store").default_retention_minutes,
        model_validate=StoredChatCompletionObject.model_validate,
        object_id_getter=lambda r: r.id,
        created_at_getter=lambda r: float(r.created),
        index_scale=CHAT_COMPLETION_INDEX_SCALE,
    )


async def list_chat_completions(
    store: StoreService,
    *,
    limit: int,
    after: str | None,
    order: str,
    predicate: Callable[[StoredChatCompletionObject], bool] | None = None,
) -> tuple[list[StoredChatCompletionObject], bool]:
    await ensure_chat_index(store)
    return await list_created_at_models(
        store,
        store_namespace=CHAT_COMPLETION_STORE_NAMESPACE,
        index_namespace=CHAT_COMPLETION_INDEX_NAMESPACE,
        limit=limit,
        after=after,
        model_validate=StoredChatCompletionObject.model_validate,
        object_id_getter=lambda r: r.id,
        created_at_getter=lambda r: float(r.created),
        index_scale=CHAT_COMPLETION_INDEX_SCALE,
        reverse=(order == "desc"),
        predicate=predicate,
    )


async def get_chat_completion(
    store: StoreService,
    completion_id: str,
) -> StoredChatCompletionObject:
    stored = await store.get(namespace=CHAT_COMPLETION_STORE_NAMESPACE, key=completion_id)
    if stored is None:
        raise NotFoundError(f"Chat completion '{completion_id}' not found.")
    return StoredChatCompletionObject.model_validate(stored)


async def update_stored_completion_metadata(
    store: StoreService,
    *,
    completion_id: str,
    metadata: dict[str, str],
) -> StoredChatCompletionObject:
    completion = (await get_chat_completion(store, completion_id)).model_copy(
        update={"metadata": metadata}
    )
    retention = GetSettings("frontend.store").default_retention_minutes
    await store.set(
        namespace=CHAT_COMPLETION_STORE_NAMESPACE,
        key=completion_id,
        value=completion.model_dump(),
        retention=retention,
    )
    await upsert_created_at_index(
        store,
        index_namespace=CHAT_COMPLETION_INDEX_NAMESPACE,
        object_id=completion.id,
        created_at=float(completion.created),
        retention=retention,
        scale=CHAT_COMPLETION_INDEX_SCALE,
    )

    stored_messages = await store.get(namespace=CHAT_COMPLETION_MESSAGES_NAMESPACE, key=completion_id)
    if stored_messages is not None:
        await store.set(
            namespace=CHAT_COMPLETION_MESSAGES_NAMESPACE,
            key=completion_id,
            value=stored_messages,
            retention=retention,
        )
    return completion


async def delete_chat_completion_state(
    store: StoreService,
    completion_id: str,
) -> bool:
    stored = await store.get(namespace=CHAT_COMPLETION_STORE_NAMESPACE, key=completion_id)
    if stored is not None:
        completion = StoredChatCompletionObject.model_validate(stored)
        await store.delete(
            namespace=CHAT_COMPLETION_INDEX_NAMESPACE,
            key=build_created_at_index_key(
                completion.created,
                completion.id,
                scale=CHAT_COMPLETION_INDEX_SCALE,
            ),
        )
    completion_deleted = await store.delete(
        namespace=CHAT_COMPLETION_STORE_NAMESPACE,
        key=completion_id,
    )
    messages_deleted = await store.delete(
        namespace=CHAT_COMPLETION_MESSAGES_NAMESPACE,
        key=completion_id,
    )
    return bool(completion_deleted or messages_deleted)


async def get_chat_completion_messages(
    store: StoreService,
    completion_id: str,
) -> list[dict]:
    await get_chat_completion(store, completion_id)
    messages = await store.get(namespace=CHAT_COMPLETION_MESSAGES_NAMESPACE, key=completion_id)
    return messages if isinstance(messages, list) else []
