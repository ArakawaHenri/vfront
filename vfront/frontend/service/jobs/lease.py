"""Shared lease management for background job ownership."""

from __future__ import annotations

import asyncio
import time

from fastapiex.di import BaseService, Require, Service

from vfront.frontend.service.jobs.namespaces import BATCH_LEASE_NAMESPACE, RESPONSE_LEASE_NAMESPACE
from vfront.frontend.service.store.lmdb import StoreMutation, StoreService
from vfront.protocol.job import JobLease


@Service("job_lease_service")
class JobLeaseService(BaseService):
    """Atomic claim/heartbeat/release over LMDB-backed job leases."""

    def __init__(self, store: StoreService) -> None:
        self._store = store
        self._lock = asyncio.Lock()

    @classmethod
    async def create(
        cls,
        store: StoreService = Require("store_service"),  # type: ignore[assignment]
    ) -> JobLeaseService:
        return cls(store=store)

    async def claim_response(
        self, response_id: str, owner_id: str, lease_seconds: int
    ) -> JobLease | None:
        return await self._claim(RESPONSE_LEASE_NAMESPACE, response_id, owner_id, lease_seconds)

    async def claim_batch(
        self, batch_id: str, owner_id: str, lease_seconds: int
    ) -> JobLease | None:
        return await self._claim(BATCH_LEASE_NAMESPACE, batch_id, owner_id, lease_seconds)

    async def heartbeat_response(
        self, response_id: str, owner_id: str, lease_seconds: int
    ) -> JobLease | None:
        return await self._heartbeat(RESPONSE_LEASE_NAMESPACE, response_id, owner_id, lease_seconds)

    async def heartbeat_batch(
        self, batch_id: str, owner_id: str, lease_seconds: int
    ) -> JobLease | None:
        return await self._heartbeat(BATCH_LEASE_NAMESPACE, batch_id, owner_id, lease_seconds)

    async def release_response(self, response_id: str, owner_id: str) -> bool:
        return await self._release(RESPONSE_LEASE_NAMESPACE, response_id, owner_id)

    async def release_batch(self, batch_id: str, owner_id: str) -> bool:
        return await self._release(BATCH_LEASE_NAMESPACE, batch_id, owner_id)

    async def clear_response(self, response_id: str) -> bool:
        return await self._clear(RESPONSE_LEASE_NAMESPACE, response_id)

    async def clear_batch(self, batch_id: str) -> bool:
        return await self._clear(BATCH_LEASE_NAMESPACE, batch_id)

    async def _claim(
        self,
        namespace: str,
        job_id: str,
        owner_id: str,
        lease_seconds: int,
    ) -> JobLease | None:
        async with self._lock:
            return await self._store.mutate(
                namespace,
                lambda mutation: self._claim_mutation(
                    mutation,
                    job_id,
                    owner_id,
                    lease_seconds,
                ),
            )

    async def _heartbeat(
        self,
        namespace: str,
        job_id: str,
        owner_id: str,
        lease_seconds: int,
    ) -> JobLease | None:
        async with self._lock:
            return await self._store.mutate(
                namespace,
                lambda mutation: self._heartbeat_mutation(
                    mutation,
                    job_id,
                    owner_id,
                    lease_seconds,
                ),
            )

    async def _release(self, namespace: str, job_id: str, owner_id: str) -> bool:
        async with self._lock:
            return await self._store.mutate(
                namespace,
                lambda mutation: self._release_mutation(mutation, job_id, owner_id),
            )

    async def _clear(self, namespace: str, job_id: str) -> bool:
        async with self._lock:
            return await self._store.mutate(
                namespace,
                lambda mutation: mutation.delete(job_id),
            )

    @staticmethod
    def _claim_mutation(
        mutation: StoreMutation,
        job_id: str,
        owner_id: str,
        lease_seconds: int,
    ) -> JobLease | None:
        now = time.time()
        current = mutation.get(job_id)
        lease = JobLease.model_validate(current) if current is not None else None
        if lease is not None and lease.lease_expires_at > now and lease.owner_id != owner_id:
            return None

        attempt = 0 if lease is None else lease.attempt
        new_lease = JobLease(
            job_id=job_id,
            owner_id=owner_id,
            lease_expires_at=now + lease_seconds,
            heartbeat_at=now,
            attempt=attempt + 1,
        )
        mutation.set(job_id, new_lease.model_dump())
        return new_lease

    @staticmethod
    def _heartbeat_mutation(
        mutation: StoreMutation,
        job_id: str,
        owner_id: str,
        lease_seconds: int,
    ) -> JobLease | None:
        current = mutation.get(job_id)
        if current is None:
            return None

        lease = JobLease.model_validate(current)
        if lease.owner_id != owner_id:
            return None

        now = time.time()
        updated = lease.model_copy(
            update={
                "lease_expires_at": now + lease_seconds,
                "heartbeat_at": now,
            }
        )
        mutation.set(job_id, updated.model_dump())
        return updated

    @staticmethod
    def _release_mutation(
        mutation: StoreMutation,
        job_id: str,
        owner_id: str,
    ) -> bool:
        current = mutation.get(job_id)
        if current is None:
            return False

        lease = JobLease.model_validate(current)
        if lease.owner_id != owner_id:
            return False
        return mutation.delete(job_id)
