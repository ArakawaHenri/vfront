"""Background job runner for responses and batches."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from fastapiex.di import BaseService, Require, Service
from fastapiex.settings import GetSettings

from vfront.frontend.service.engine.router import EngineRouter
from vfront.frontend.service.file_storage.main import FileStorageService
from vfront.frontend.service.jobs.lease import JobLeaseService
from vfront.frontend.service.jobs.runtime import should_run_job_runner
from vfront.frontend.service.jobs.store import JobStoreService
from vfront.frontend.service.mcp.client import MCPClientManager
from vfront.frontend.service.persistence.response_store import (
    RESPONSE_STORE_NAMESPACE,
    store_response_state,
)
from vfront.frontend.service.store.lmdb import StoreService
from vfront.frontend.service.tool_parsing.service import ToolParsingService
from vfront.protocol.batch import Batch, BatchError, BatchErrors
from vfront.protocol.job import BatchExecution, ResponseExecution
from vfront.protocol.response import ResponseObject

if TYPE_CHECKING:
    from vfront.frontend.service.jobs.settings import JobSettings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ActiveTask:
    kind: str
    job_id: str
    task: asyncio.Task


@dataclass(slots=True)
class _ClaimCandidate:
    kind: Literal["response", "batch"]
    job_id: str
    status: str
    created_at: float
    updated_at: float


@Service("job_runner_service", eager=True)
class JobRunnerService(BaseService):
    """Embedded or standalone background job runner."""

    def __init__(
        self,
        *,
        engine_router: EngineRouter,
        store: StoreService,
        file_storage: FileStorageService,
        mcp_manager: MCPClientManager,
        tool_parsing_service: ToolParsingService,
        job_store: JobStoreService,
        lease_service: JobLeaseService,
        settings: JobSettings,
    ) -> None:
        self._engine_router = engine_router
        self._store = store
        self._file_storage = file_storage
        self._mcp_manager = mcp_manager
        self._tool_parsing_service = tool_parsing_service
        self._job_store = job_store
        self._lease_service = lease_service
        self._settings = settings
        self._owner_id = settings.runner_id or f"runner_{uuid.uuid4().hex[:12]}"
        self._tasks: dict[tuple[str, str], _ActiveTask] = {}
        self._loop_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._wakeup = asyncio.Event()
        self._enabled = should_run_job_runner(settings)

    @classmethod
    async def create(
        cls,
        engine_router: EngineRouter = Require("engine_router"),  # type: ignore[assignment]
        store: StoreService = Require("store_service"),  # type: ignore[assignment]
        file_storage: FileStorageService = Require("file_storage_service"),  # type: ignore[assignment]
        mcp_manager: MCPClientManager = Require("mcp_client_manager"),  # type: ignore[assignment]
        tool_parsing_service: ToolParsingService = Require("tool_parsing_service"),  # type: ignore[assignment]
        job_store: JobStoreService = Require("job_store_service"),  # type: ignore[assignment]
        lease_service: JobLeaseService = Require("job_lease_service"),  # type: ignore[assignment]
    ) -> JobRunnerService:
        settings = GetSettings("frontend.jobs")
        instance = cls(
            engine_router=engine_router,
            store=store,
            file_storage=file_storage,
            mcp_manager=mcp_manager,
            tool_parsing_service=tool_parsing_service,
            job_store=job_store,
            lease_service=lease_service,
            settings=settings,
        )
        if instance._enabled:
            await instance._heartbeat_and_prune()
            await instance._claim_pending()
            instance._loop_task = asyncio.create_task(instance._run_loop())
        return instance

    @classmethod
    async def destroy(cls, instance: JobRunnerService) -> None:
        instance._stop.set()
        if instance._loop_task is not None:
            instance._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await instance._loop_task
        for active in list(instance._tasks.values()):
            active.task.cancel()
        for active in list(instance._tasks.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await active.task
        await instance._release_owned_work()

    async def request_cancel(self, kind: str, job_id: str) -> None:
        active = self._tasks.get((kind, job_id))
        if active is not None:
            active.task.cancel()
        self._wakeup.set()

    async def notify(self) -> None:
        if self._enabled:
            self._wakeup.set()

    async def _release_owned_work(self) -> None:
        now = time.time()
        from vfront.frontend.api.v1.batches import _BATCH_NAMESPACE, _store_batch_state

        for response_execution in await self._job_store.list_response_executions():
            if (
                response_execution.owner_id != self._owner_id
                or response_execution.status != "running"
            ):
                continue
            response_updates: dict[str, object] = {"owner_id": None}
            if response_execution.cancel_requested:
                response_updates.update(
                    {
                        "status": "cancelled",
                        "finished_at": response_execution.finished_at or now,
                    }
                )
                stored = await self._store.get(
                    namespace=RESPONSE_STORE_NAMESPACE,
                    key=response_execution.response_id,
                )
                if stored is not None:
                    response = ResponseObject.model_validate(stored).model_copy(
                        update={
                            "status": "cancelled",
                            "completed_at": response_execution.finished_at or now,
                        }
                    )
                    await store_response_state(self._store, response)
                await self._job_store.delete_response_spec(response_execution.response_id)
            else:
                response_updates["status"] = "queued"
            await self._job_store.update_response_execution(
                response_execution.response_id,
                **response_updates,
            )
            await self._lease_service.release_response(
                response_execution.response_id,
                self._owner_id,
            )

        for batch_execution in await self._job_store.list_batch_executions():
            if batch_execution.owner_id != self._owner_id or batch_execution.status != "running":
                continue
            batch_updates: dict[str, object] = {"owner_id": None}
            if batch_execution.cancel_requested:
                batch_updates.update(
                    {
                        "status": "cancelled",
                        "finished_at": batch_execution.finished_at or now,
                    }
                )
                stored = await self._store.get(namespace=_BATCH_NAMESPACE, key=batch_execution.batch_id)
                if stored is not None:
                    batch = Batch.model_validate(stored).model_copy(
                        update={
                            "status": "cancelled",
                            "cancelled_at": int(batch_execution.finished_at or now),
                        }
                    )
                    await _store_batch_state(self._store, batch)
                await self._job_store.delete_batch_spec(batch_execution.batch_id)
            else:
                batch_updates["status"] = "queued"
            await self._job_store.update_batch_execution(
                batch_execution.batch_id,
                **batch_updates,
            )
            await self._lease_service.release_batch(batch_execution.batch_id, self._owner_id)

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._heartbeat_and_prune()
                await self._claim_pending()
            except Exception:
                logger.exception("Job runner loop error")
            try:
                self._wakeup.clear()
                await asyncio.wait_for(
                    self._wakeup.wait(),
                    timeout=self._settings.poll_interval_seconds,
                )
            except TimeoutError:
                continue

    async def _heartbeat_and_prune(self) -> None:
        for key, active in list(self._tasks.items()):
            if active.task.done():
                self._tasks.pop(key, None)
                continue

            if active.kind == "response":
                response_execution = await self._job_store.get_response_execution(active.job_id)
                if response_execution is not None and response_execution.cancel_requested:
                    active.task.cancel()
                await self._lease_service.heartbeat_response(
                    active.job_id,
                    self._owner_id,
                    self._settings.lease_seconds,
                )
            else:
                batch_execution = await self._job_store.get_batch_execution(active.job_id)
                if batch_execution is not None and batch_execution.cancel_requested:
                    active.task.cancel()
                await self._lease_service.heartbeat_batch(
                    active.job_id,
                    self._owner_id,
                    self._settings.lease_seconds,
                )

    async def _claim_pending(self) -> None:
        available = self._settings.max_parallel_jobs - len(self._tasks)
        if available <= 0:
            return

        candidates: list[_ClaimCandidate] = []
        for response_execution in await self._job_store.list_response_executions():
            if self._is_claimable_response(response_execution):
                candidates.append(
                    _ClaimCandidate(
                        kind="response",
                        job_id=response_execution.response_id,
                        status=response_execution.status,
                        created_at=response_execution.created_at,
                        updated_at=response_execution.updated_at,
                    )
                )

        for batch_execution in await self._job_store.list_batch_executions():
            if self._is_claimable_batch(batch_execution):
                candidates.append(
                    _ClaimCandidate(
                        kind="batch",
                        job_id=batch_execution.batch_id,
                        status=batch_execution.status,
                        created_at=batch_execution.created_at,
                        updated_at=batch_execution.updated_at,
                    )
                )

        candidates.sort(
            key=lambda candidate: (
                0 if candidate.status == "running" else 1,
                -candidate.updated_at,
                -candidate.created_at,
            )
        )

        for candidate in candidates:
            if available <= 0:
                break

            if candidate.kind == "response":
                loaded_response_execution = await self._job_store.get_response_execution(
                    candidate.job_id
                )
                if loaded_response_execution is None or not self._is_claimable_response(
                    loaded_response_execution
                ):
                    continue
                if await self._claim_response(loaded_response_execution):
                    available -= 1
                continue

            loaded_batch_execution = await self._job_store.get_batch_execution(candidate.job_id)
            if loaded_batch_execution is None or not self._is_claimable_batch(
                loaded_batch_execution
            ):
                continue
            if await self._claim_batch(loaded_batch_execution):
                available -= 1

    def _is_claimable_response(self, execution: ResponseExecution) -> bool:
        if ("response", execution.response_id) in self._tasks or execution.cancel_requested:
            return False
        return execution.status in {"queued", "running"}

    def _is_claimable_batch(self, execution: BatchExecution) -> bool:
        if ("batch", execution.batch_id) in self._tasks or execution.cancel_requested:
            return False
        return execution.status in {"queued", "running"}

    async def _claim_response(self, execution: ResponseExecution) -> bool:
        lease = await self._lease_service.claim_response(
            execution.response_id,
            self._owner_id,
            self._settings.lease_seconds,
        )
        if lease is None:
            return False

        now = time.time()
        try:
            await self._job_store.update_response_execution(
                execution.response_id,
                status="running",
                owner_id=self._owner_id,
                attempt=lease.attempt,
                started_at=execution.started_at or now,
            )
            task = asyncio.create_task(self._run_response(execution.response_id))
            self._tasks[("response", execution.response_id)] = _ActiveTask(
                kind="response",
                job_id=execution.response_id,
                task=task,
            )
            return True
        except BaseException:
            await self._lease_service.release_response(execution.response_id, self._owner_id)
            raise

    async def _claim_batch(self, execution: BatchExecution) -> bool:
        lease = await self._lease_service.claim_batch(
            execution.batch_id,
            self._owner_id,
            self._settings.lease_seconds,
        )
        if lease is None:
            return False

        now = time.time()
        try:
            await self._job_store.update_batch_execution(
                execution.batch_id,
                status="running",
                owner_id=self._owner_id,
                attempt=lease.attempt,
                started_at=execution.started_at or now,
            )
            task = asyncio.create_task(self._run_batch(execution.batch_id))
            self._tasks[("batch", execution.batch_id)] = _ActiveTask(
                kind="batch",
                job_id=execution.batch_id,
                task=task,
            )
            return True
        except BaseException:
            await self._lease_service.release_batch(execution.batch_id, self._owner_id)
            raise

    async def _run_response(self, response_id: str) -> None:
        from vfront.frontend.api.v1 import responses as response_api

        spec = await self._job_store.get_response_spec(response_id)
        if spec is None:
            await self._job_store.update_response_execution(
                response_id,
                status="failed",
                owner_id=None,
                finished_at=time.time(),
                last_error="missing response job spec",
            )
            await self._lease_service.release_response(response_id, self._owner_id)
            return

        try:
            await response_api._background_generate(
                spec=spec,
                engine_router=self._engine_router,
                store=self._store,
                mcp_manager=self._mcp_manager,
                tool_parsing_service=self._tool_parsing_service,
            )
        except asyncio.CancelledError:
            if not self._stop.is_set():
                stored = await self._store.get(
                    namespace=RESPONSE_STORE_NAMESPACE,
                    key=response_id,
                )
                if stored is not None:
                    response = ResponseObject.model_validate(stored).model_copy(
                        update={"status": "cancelled", "completed_at": time.time()}
                    )
                    await store_response_state(self._store, response)
                await self._job_store.update_response_execution(
                    response_id,
                    status="cancelled",
                    owner_id=None,
                    finished_at=time.time(),
                )
                await self._job_store.delete_response_spec(response_id)
            raise
        except Exception as exc:
            await self._job_store.update_response_execution(
                response_id,
                status="failed",
                owner_id=None,
                finished_at=time.time(),
                last_error=str(exc),
            )
            await self._job_store.delete_response_spec(response_id)
            logger.exception("Response job failed in runner: %s", response_id)
        else:
            await self._job_store.update_response_execution(
                response_id,
                status="completed",
                owner_id=None,
                finished_at=time.time(),
                last_error=None,
            )
            await self._job_store.delete_response_spec(response_id)
        finally:
            await self._lease_service.release_response(response_id, self._owner_id)

    async def _run_batch(self, batch_id: str) -> None:
        from vfront.frontend.api.v1.batches import (
            _BATCH_NAMESPACE,
            _process_batch,
            _store_batch_state,
        )

        spec = await self._job_store.get_batch_spec(batch_id)
        if spec is None:
            await self._job_store.update_batch_execution(
                batch_id,
                status="failed",
                owner_id=None,
                finished_at=time.time(),
                last_error="missing batch job spec",
            )
            await self._lease_service.release_batch(batch_id, self._owner_id)
            return

        try:
            await _process_batch(
                spec=spec,
                engine_router=self._engine_router,
                store=self._store,
                file_storage=self._file_storage,
                mcp_manager=self._mcp_manager,
                tool_parsing_service=self._tool_parsing_service,
            )
        except asyncio.CancelledError:
            if not self._stop.is_set():
                stored = await self._store.get(namespace=_BATCH_NAMESPACE, key=batch_id)
                if stored is not None:
                    batch = Batch.model_validate(stored).model_copy(
                        update={"status": "cancelled", "cancelled_at": int(time.time())}
                    )
                    await _store_batch_state(self._store, batch)
                await self._job_store.update_batch_execution(
                    batch_id,
                    status="cancelled",
                    owner_id=None,
                    finished_at=time.time(),
                )
                await self._job_store.delete_batch_spec(batch_id)
            raise
        except Exception as exc:
            stored = await self._store.get(namespace=_BATCH_NAMESPACE, key=batch_id)
            if stored is not None:
                batch = Batch.model_validate(stored).model_copy(
                    update={
                        "status": "failed",
                        "failed_at": int(time.time()),
                        "errors": BatchErrors(
                            data=[
                                BatchError(
                                    code="server_error", message="Internal batch processing error"
                                )
                            ]
                        ),
                    }
                )
                await _store_batch_state(self._store, batch)
            await self._job_store.update_batch_execution(
                batch_id,
                status="failed",
                owner_id=None,
                finished_at=time.time(),
                last_error=str(exc),
            )
            await self._job_store.delete_batch_spec(batch_id)
            logger.exception("Batch job failed in runner: %s", batch_id)
        else:
            await self._job_store.update_batch_execution(
                batch_id,
                status="completed",
                owner_id=None,
                finished_at=time.time(),
                last_error=None,
            )
            await self._job_store.delete_batch_spec(batch_id)
        finally:
            await self._lease_service.release_batch(batch_id, self._owner_id)
