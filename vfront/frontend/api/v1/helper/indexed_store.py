"""Helpers for created_at-indexed persistence in API resources."""

from __future__ import annotations

from collections.abc import Callable

from vfront.frontend.service.store.lmdb import StoreService


def build_created_at_index_key(
    created_at: int | float,
    object_id: str,
    *,
    scale: int = 1,
) -> str:
    """Build a lexicographically sortable created_at index key."""
    normalized = max(int(created_at * scale), 0)
    return f"{normalized:020d}:{object_id}"


async def upsert_created_at_index(
    store: StoreService,
    *,
    index_namespace: str,
    object_id: str,
    created_at: int | float,
    retention: int,
    scale: int = 1,
) -> None:
    """Create or update a created_at index entry."""
    await store.set(
        namespace=index_namespace,
        key=build_created_at_index_key(created_at, object_id, scale=scale),
        value=object_id,
        retention=retention,
    )


async def store_created_at_indexed_model[T](
    store: StoreService,
    *,
    store_namespace: str,
    index_namespace: str,
    model: T,
    retention: int,
    object_id_getter: Callable[[T], str],
    created_at_getter: Callable[[T], int | float],
    payload_getter: Callable[[T], object],
    index_scale: int = 1,
) -> None:
    """Persist a model together with its created_at index entry."""
    object_id = object_id_getter(model)
    await store.set(
        namespace=store_namespace,
        key=object_id,
        value=payload_getter(model),
        retention=retention,
    )
    await upsert_created_at_index(
        store,
        index_namespace=index_namespace,
        object_id=object_id,
        created_at=created_at_getter(model),
        retention=retention,
        scale=index_scale,
    )


async def ensure_created_at_index[T](
    store: StoreService,
    *,
    store_namespace: str,
    index_namespace: str,
    retention: int,
    model_validate: Callable[[object], T],
    object_id_getter: Callable[[T], str],
    created_at_getter: Callable[[T], int | float],
    index_scale: int = 1,
) -> None:
    """Backfill a created_at index lazily if it does not exist yet."""
    existing_index, _ = await store.list_page(
        namespace=index_namespace,
        limit=1,
        reverse=True,
    )
    if existing_index:
        return

    object_ids = await store.list_keys(namespace=store_namespace)
    if not object_ids:
        return

    for object_id in object_ids:
        stored = await store.get(namespace=store_namespace, key=object_id)
        if stored is None:
            continue
        model = model_validate(stored)
        await upsert_created_at_index(
            store,
            index_namespace=index_namespace,
            object_id=object_id_getter(model),
            created_at=created_at_getter(model),
            retention=retention,
            scale=index_scale,
        )


async def list_created_at_models[T](
    store: StoreService,
    *,
    store_namespace: str,
    index_namespace: str,
    limit: int,
    after: str | None,
    model_validate: Callable[[object], T],
    object_id_getter: Callable[[T], str],
    created_at_getter: Callable[[T], int | float],
    index_scale: int = 1,
    reverse: bool = True,
    predicate: Callable[[T], bool] | None = None,
) -> tuple[list[T], bool]:
    """List models ordered by their created_at index.

    When *predicate* is provided, only models for which ``predicate(model)``
    returns ``True`` are counted and returned.  The scan continues across index
    pages until ``limit + 1`` matching models are collected or the index is
    exhausted, so ``has_more`` and ``after``-based cursor pagination remain
    correct even when filtering is active.
    """
    after_index_key: str | None = None
    if after is not None:
        after_data = await store.get(namespace=store_namespace, key=after)
        if after_data is None:
            return [], False
        after_model = model_validate(after_data)
        after_index_key = build_created_at_index_key(
            created_at_getter(after_model),
            object_id_getter(after_model),
            scale=index_scale,
        )

    models: list[T] = []
    scan_after = after_index_key
    batch_size = min(max(limit * 2, 50), 500)

    while len(models) <= limit:
        page, has_more_page = await store.list_page(
            namespace=index_namespace,
            after=scan_after,
            limit=batch_size,
            reverse=reverse,
        )
        if not page:
            break

        for _index_key, object_id in page:
            stored = await store.get(namespace=store_namespace, key=str(object_id))
            if stored is None:
                continue
            model = model_validate(stored)
            if predicate is None or predicate(model):
                models.append(model)
                if len(models) > limit:
                    break

        if len(models) > limit or not has_more_page:
            break
        scan_after = page[-1][0]

    has_more = len(models) > limit
    return models[:limit], has_more
