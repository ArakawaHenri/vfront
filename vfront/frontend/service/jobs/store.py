"""Persistence helpers for background job specs, execution state, and leases."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from fastapiex.di import BaseService, Require, Service
from fastapiex.settings import GetSettings

from vfront.frontend.service.jobs.namespaces import (
    BATCH_EXEC_NAMESPACE,
    BATCH_JOB_NAMESPACE,
    RESPONSE_EXEC_NAMESPACE,
    RESPONSE_JOB_NAMESPACE,
)
from vfront.frontend.service.store.lmdb import StoreMutation, StoreService
from vfront.protocol.job import (
    BatchExecution,
    BatchJobSpec,
    ResponseExecution,
    ResponseJobSpec,
)

TExecution = TypeVar("TExecution", ResponseExecution, BatchExecution)
_TERMINAL_EXECUTION_STATUSES = {"completed", "failed", "cancelled"}


@dataclass(slots=True)
class ResponseCancelUpdate:
    previous_status: str
    execution: ResponseExecution


@dataclass(slots=True)
class BatchCancelUpdate:
    previous_status: str
    execution: BatchExecution


@Service("job_store_service")
class JobStoreService(BaseService):
    """High-level store API for durable background jobs."""

    def __init__(self, store: StoreService) -> None:
        self._store = store

    @classmethod
    async def create(
        cls,
        store: StoreService = Require("store_service"),  # type: ignore[assignment]
    ) -> JobStoreService:
        return cls(store=store)

    async def put_response_spec(self, spec: ResponseJobSpec) -> None:
        await self._store.set(
            namespace=RESPONSE_JOB_NAMESPACE,
            key=spec.response_id,
            value=spec.model_dump(),
            retention=GetSettings("frontend.store").default_retention_minutes,
        )

    async def get_response_spec(self, response_id: str) -> ResponseJobSpec | None:
        data = await self._store.get(namespace=RESPONSE_JOB_NAMESPACE, key=response_id)
        if data is None:
            return None
        return ResponseJobSpec.model_validate(data)

    async def delete_response_spec(self, response_id: str) -> bool:
        return await self._store.delete(namespace=RESPONSE_JOB_NAMESPACE, key=response_id)

    async def put_batch_spec(self, spec: BatchJobSpec) -> None:
        await self._store.set(
            namespace=BATCH_JOB_NAMESPACE,
            key=spec.batch_id,
            value=spec.model_dump(),
            retention=GetSettings("frontend.store").default_retention_minutes,
        )

    async def get_batch_spec(self, batch_id: str) -> BatchJobSpec | None:
        data = await self._store.get(namespace=BATCH_JOB_NAMESPACE, key=batch_id)
        if data is None:
            return None
        return BatchJobSpec.model_validate(data)

    async def delete_batch_spec(self, batch_id: str) -> bool:
        return await self._store.delete(namespace=BATCH_JOB_NAMESPACE, key=batch_id)

    async def put_response_execution(self, execution: ResponseExecution) -> None:
        await self._store.set(
            namespace=RESPONSE_EXEC_NAMESPACE,
            key=execution.response_id,
            value=execution.model_dump(),
            retention=GetSettings("frontend.store").default_retention_minutes,
        )

    async def get_response_execution(self, response_id: str) -> ResponseExecution | None:
        data = await self._store.get(namespace=RESPONSE_EXEC_NAMESPACE, key=response_id)
        if data is None:
            return None
        return ResponseExecution.model_validate(data)

    async def update_response_execution(
        self, response_id: str, **updates: object
    ) -> ResponseExecution | None:
        return await self._update_execution(
            response_id,
            getter=self.get_response_execution,
            putter=self.put_response_execution,
            **updates,
        )

    async def list_response_executions(self) -> list[ResponseExecution]:
        executions: list[ResponseExecution] = []
        for response_id in await self._store.list_keys(namespace=RESPONSE_EXEC_NAMESPACE):
            execution = await self.get_response_execution(response_id)
            if execution is not None:
                executions.append(execution)
        executions.sort(key=lambda item: item.created_at)
        return executions

    async def delete_response_execution(self, response_id: str) -> bool:
        return await self._store.delete(namespace=RESPONSE_EXEC_NAMESPACE, key=response_id)

    async def request_cancel_response_execution(
        self,
        response_id: str,
    ) -> ResponseCancelUpdate | None:
        return await self._store.mutate(
            RESPONSE_EXEC_NAMESPACE,
            lambda mutation: self._request_cancel_response_execution_mutation(
                mutation,
                response_id,
            ),
        )

    async def put_batch_execution(self, execution: BatchExecution) -> None:
        await self._store.set(
            namespace=BATCH_EXEC_NAMESPACE,
            key=execution.batch_id,
            value=execution.model_dump(),
            retention=GetSettings("frontend.store").default_retention_minutes,
        )

    async def get_batch_execution(self, batch_id: str) -> BatchExecution | None:
        data = await self._store.get(namespace=BATCH_EXEC_NAMESPACE, key=batch_id)
        if data is None:
            return None
        return BatchExecution.model_validate(data)

    async def update_batch_execution(
        self, batch_id: str, **updates: object
    ) -> BatchExecution | None:
        return await self._update_execution(
            batch_id,
            getter=self.get_batch_execution,
            putter=self.put_batch_execution,
            **updates,
        )

    async def list_batch_executions(self) -> list[BatchExecution]:
        executions: list[BatchExecution] = []
        for batch_id in await self._store.list_keys(namespace=BATCH_EXEC_NAMESPACE):
            execution = await self.get_batch_execution(batch_id)
            if execution is not None:
                executions.append(execution)
        executions.sort(key=lambda item: item.created_at)
        return executions

    async def delete_batch_execution(self, batch_id: str) -> bool:
        return await self._store.delete(namespace=BATCH_EXEC_NAMESPACE, key=batch_id)

    async def request_cancel_batch_execution(
        self,
        batch_id: str,
    ) -> BatchCancelUpdate | None:
        return await self._store.mutate(
            BATCH_EXEC_NAMESPACE,
            lambda mutation: self._request_cancel_batch_execution_mutation(
                mutation,
                batch_id,
            ),
        )

    async def _update_execution(
        self,
        key: str,
        *,
        getter: Callable[[str], Awaitable[TExecution | None]],
        putter: Callable[[TExecution], Awaitable[None]],
        **updates: object,
    ) -> TExecution | None:
        execution = await getter(key)
        if execution is None:
            return None
        updated_execution = execution.model_copy(update={**updates, "updated_at": time.time()})
        await putter(updated_execution)
        return updated_execution

    @staticmethod
    def _request_cancel_response_execution_mutation(
        mutation: StoreMutation,
        response_id: str,
    ) -> ResponseCancelUpdate | None:
        current = mutation.get(response_id)
        if current is None:
            return None

        execution = ResponseExecution.model_validate(current)
        if execution.status in _TERMINAL_EXECUTION_STATUSES:
            return ResponseCancelUpdate(
                previous_status=execution.status,
                execution=execution,
            )

        now = time.time()
        updates: dict[str, object] = {"cancel_requested": True, "updated_at": now}
        if execution.status == "queued":
            updates.update({"status": "cancelled", "finished_at": now})
        updated_execution = execution.model_copy(update=updates)
        mutation.set(response_id, updated_execution.model_dump())
        return ResponseCancelUpdate(
            previous_status=execution.status,
            execution=updated_execution,
        )

    @staticmethod
    def _request_cancel_batch_execution_mutation(
        mutation: StoreMutation,
        batch_id: str,
    ) -> BatchCancelUpdate | None:
        current = mutation.get(batch_id)
        if current is None:
            return None

        execution = BatchExecution.model_validate(current)
        if execution.status in _TERMINAL_EXECUTION_STATUSES:
            return BatchCancelUpdate(
                previous_status=execution.status,
                execution=execution,
            )

        now = time.time()
        updates: dict[str, object] = {"cancel_requested": True, "updated_at": now}
        if execution.status == "queued":
            updates.update({"status": "cancelled", "finished_at": now})
        updated_execution = execution.model_copy(update=updates)
        mutation.set(batch_id, updated_execution.model_dump())
        return BatchCancelUpdate(
            previous_status=execution.status,
            execution=updated_execution,
        )
