"""LMDB-backed key-value store service.

Provides an async interface over a persistent LMDB environment with:
- Namespace isolation via named LMDB sub-databases.
- Optional per-entry expiry (TTL in minutes).
- A periodic cleanup loop to purge expired entries.
- CBOR2 serialisation for arbitrary Python values.

Adapted from neo_tx_classifier's StoreService, using zlmdb for LMDB bindings.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar

import cbor2
import zlmdb.lmdb as lmdb
from fastapiex.di import BaseService, Service
from fastapiex.settings import GetSettings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_EXPIRY_STRUCT = struct.Struct(">q")
_BUCKET_STRUCT = struct.Struct(">q")
_INT64_MAX = 2**63 - 1
_META_DB_NAME = b"__meta__"
_RESERVED_PREFIXES = ("__exp__", "__expmeta__", "__meta__")


class StoreMutation:
    """Single-namespace atomic mutation handle."""

    def __init__(
        self,
        *,
        store: StoreService,
        txn: Any,
        db: Any,
        exp_db: Any,
        expmeta_db: Any,
    ) -> None:
        self._store = store
        self._txn = txn
        self._db = db
        self._exp_db = exp_db
        self._expmeta_db = expmeta_db

    def get(self, key: str) -> object | None:
        key_bytes = self._store._esc_key(key)
        return self._store._txn_get(self._txn, self._db, key_bytes)

    def set(self, key: str, value: object, *, retention: int | None = None) -> None:
        if retention is not None and retention < 1:
            raise ValueError("retention must be a positive integer (minutes)")

        expire_ts = 0
        if retention is not None:
            expire_ts = int(time.time() + retention * 60)
            if expire_ts > _INT64_MAX:
                raise ValueError("retention too large; expiry overflows int64")

        key_bytes = self._store._esc_key(key)
        self._store._validate_key_size(key_bytes)
        self._store._txn_put(
            self._txn,
            self._db,
            self._exp_db,
            self._expmeta_db,
            key_bytes,
            value,
            expire_ts,
        )

    def delete(self, key: str) -> bool:
        key_bytes = self._store._esc_key(key)
        return self._store._txn_delete(
            self._txn,
            self._db,
            self._exp_db,
            self._expmeta_db,
            key_bytes,
        )


@Service("store_service", eager=True)
class StoreService(BaseService):
    """LMDB-backed key-value store with namespace isolation and TTL support."""

    @classmethod
    async def create(cls) -> StoreService:
        cfg = GetSettings("frontend.store")
        service = StoreService(
            path=str(cfg.path),
            map_size_mb=cfg.map_size_mb,
            max_dbs=cfg.max_dbs,
            max_readers=cfg.max_readers,
            sync=cfg.sync,
            metasync=cfg.metasync,
            writemap=cfg.writemap,
            map_async=cfg.map_async,
            max_key_bytes=cfg.max_key_bytes,
            max_value_bytes=cfg.max_value_bytes,
            cleanup_interval=cfg.cleanup_interval_seconds,
            cleanup_max_deletes=cfg.cleanup_max_deletes,
        )
        service._start_cleanup()
        return service

    @classmethod
    async def destroy(cls, instance: StoreService) -> None:
        await instance.shutdown()

    def __init__(
        self,
        path: str,
        map_size_mb: int = 512,
        max_dbs: int = 64,
        max_readers: int = 256,
        sync: bool = False,
        metasync: bool = True,
        writemap: bool = True,
        map_async: bool = True,
        max_key_bytes: int = 256,
        max_value_bytes: int = 100 * 1024 * 1024,
        cleanup_interval: int = 60,
        cleanup_max_deletes: int = 1_000_000,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_key_bytes = max_key_bytes
        self._max_value_bytes = max_value_bytes
        self._cleanup_interval = cleanup_interval
        self._cleanup_max_deletes = cleanup_max_deletes

        self._env = lmdb.open(
            str(self._path),
            map_size=map_size_mb * 1024 * 1024,
            max_dbs=max_dbs,
            max_readers=max_readers,
            sync=sync,
            metasync=metasync,
            writemap=writemap,
            map_async=map_async,
            lock=True,
            readahead=False,
            meminit=False,
        )

        self._db_cache: dict[str, Any] = {}
        self._exp_cache: dict[str, Any] = {}
        self._expmeta_cache: dict[str, Any] = {}
        self._meta_db: Any = self._env.open_db(_META_DB_NAME, create=True)
        self._db_cache_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._namespace_lock = asyncio.Lock()
        self._namespaces: set[str] = set()
        self._operation_lock = asyncio.Lock()
        self._active_operations = 0
        self._operations_drained = asyncio.Event()
        self._operations_drained.set()

        self._cleanup_stop = asyncio.Event()
        self._cleanup_task: asyncio.Task | None = None
        self._closing = False
        self._closed = False

        logger.debug("StoreService initialized | path=%s", self._path)

    def _start_cleanup(self) -> None:
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Periodically purge expired entries."""
        logger.debug("Store cleanup loop started")
        try:
            while not self._cleanup_stop.is_set():
                try:
                    await self.cleanup_expired(time.time())
                except Exception:
                    logger.exception("Store cleanup error")
                try:
                    await asyncio.wait_for(
                        self._cleanup_stop.wait(),
                        timeout=self._cleanup_interval,
                    )
                except TimeoutError:
                    continue
        finally:
            logger.debug("Store cleanup loop stopped")

    async def shutdown(self) -> None:
        if self._closed:
            return
        async with self._operation_lock:
            if self._closed:
                return
            self._closing = True
        self._cleanup_stop.set()
        if self._cleanup_task is not None:
            await self._cleanup_task
        await self._operations_drained.wait()
        try:
            with self._write_lock:
                self._env.sync()
        finally:
            with self._write_lock:
                self._env.close()
        self._closed = True
        logger.debug("StoreService shutdown completed")

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def set(
        self,
        namespace: str = "default",
        key: str = "",
        value: object = "",
        retention: int | None = None,
    ) -> None:
        """Upsert a value. ``retention`` is in minutes; None means no expiry."""
        async with self._operation():
            if not key:
                raise ValueError("Store key must be non-empty")
            if retention is not None and retention < 1:
                raise ValueError("retention must be a positive integer (minutes)")

            expire_ts = 0
            if retention is not None:
                expire_ts = int(time.time() + retention * 60)
                if expire_ts > _INT64_MAX:
                    raise ValueError("retention too large; expiry overflows int64")

            ns = self._esc_namespace(namespace)
            key_bytes = self._esc_key(key)
            self._validate_key_size(key_bytes)
            await self._register_namespace(ns)
            self._put(ns, key_bytes, value, expire_ts)

    async def get(self, namespace: str = "default", key: str = "") -> object | None:
        """Return the value for *key*, or None if absent/expired."""
        async with self._operation():
            if not key:
                raise ValueError("Store key must be non-empty")
            ns = self._esc_namespace(namespace)
            key_bytes = self._esc_key(key)
            await self._register_namespace(ns)
            return self._get(ns, key_bytes)

    async def delete(self, namespace: str = "default", key: str = "") -> bool:
        """Delete a key. Returns True if it existed."""
        async with self._operation():
            if not key:
                raise ValueError("Store key must be non-empty")
            ns = self._esc_namespace(namespace)
            key_bytes = self._esc_key(key)
            await self._register_namespace(ns)
            return self._delete(ns, key_bytes)

    async def list_keys(self, namespace: str = "default") -> list[str]:
        """List all non-expired keys in a namespace."""
        async with self._operation():
            ns = self._esc_namespace(namespace)
            await self._register_namespace(ns)
            return self._list_keys(ns)

    async def list_page(
        self,
        namespace: str = "default",
        *,
        after: str | None = None,
        limit: int = 100,
        reverse: bool = False,
    ) -> tuple[list[tuple[str, object]], bool]:
        """List a lexicographically ordered page of non-expired items."""
        async with self._operation():
            if limit < 1:
                raise ValueError("limit must be >= 1")
            ns = self._esc_namespace(namespace)
            await self._register_namespace(ns)
            after_bytes = self._esc_key(after) if after is not None else None
            return self._list_page(
                ns,
                after_bytes,
                limit,
                reverse,
            )

    async def mutate(
        self,
        namespace: str = "default",
        handler: Callable[[StoreMutation], T] | None = None,
    ) -> T:
        """Run an atomic write transaction within a namespace."""
        async with self._operation():
            if handler is None:
                raise ValueError("handler is required")
            ns = self._esc_namespace(namespace)
            await self._register_namespace(ns)
            return self._mutate(ns, handler)

    def _list_keys(self, ns: str) -> list[str]:
        """Blocking: iterate all keys in a namespace data DB."""
        data_db = self._get_db(ns)
        now = time.time()
        keys: list[str] = []
        with self._env.begin(db=data_db, write=False) as txn:
            cursor = txn.cursor()
            for key_bytes, value_bytes in cursor:
                if self._decode_raw_value(value_bytes, now) is None:
                    continue
                keys.append(key_bytes.decode("utf-8"))
        return keys

    def _list_page(
        self,
        ns: str,
        after_key: bytes | None,
        limit: int,
        reverse: bool,
    ) -> tuple[list[tuple[str, object]], bool]:
        """Blocking: iterate a page of items in lexicographic key order."""
        data_db = self._get_db(ns)
        now = time.time()
        items: list[tuple[str, object]] = []

        with self._env.begin(db=data_db, write=False) as txn:
            cursor = txn.cursor()
            if not self._position_cursor(
                cursor=cursor,
                after_key=after_key,
                reverse=reverse,
            ):
                return [], False

            while True:
                key_bytes, value_bytes = cursor.item()
                value = self._decode_raw_value(value_bytes, now)
                if value is not None:
                    items.append((key_bytes.decode("utf-8"), value))
                    if len(items) > limit:
                        return items[:limit], True

                moved = cursor.prev() if reverse else cursor.next()
                if not moved:
                    break

        return items, False

    async def cleanup_expired(self, now: float | None = None) -> None:
        async with self._operation():
            if now is None:
                now = time.time()
            namespaces = self._list_namespaces()
            async with self._namespace_lock:
                for ns in self._namespaces:
                    if ns not in namespaces:
                        namespaces.append(ns)
            if not namespaces:
                return
            self._cleanup_env(namespaces, now)

    # ------------------------------------------------------------------
    # Sync helpers (called via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _put(self, namespace: str, key_bytes: bytes, value: object, expire_ts: int) -> None:
        db = self._get_db(namespace)
        exp_db = self._get_exp_db(namespace)
        expmeta_db = self._get_expmeta_db(namespace)
        with self._write_lock, self._env.begin(write=True) as txn:
            self._txn_put(txn, db, exp_db, expmeta_db, key_bytes, value, expire_ts)

    def _get(self, namespace: str, key_bytes: bytes) -> object | None:
        db = self._get_db(namespace)
        with self._env.begin(write=False) as txn:
            return self._txn_get(txn, db, key_bytes)

    def _delete(self, namespace: str, key_bytes: bytes) -> bool:
        db = self._get_db(namespace)
        expmeta_db = self._get_expmeta_db(namespace)
        exp_db = self._get_exp_db(namespace)
        with self._write_lock, self._env.begin(write=True) as txn:
            return self._txn_delete(txn, db, exp_db, expmeta_db, key_bytes)

    def _mutate(self, namespace: str, handler: Callable[[StoreMutation], T]) -> T:
        db = self._get_db(namespace)
        exp_db = self._get_exp_db(namespace)
        expmeta_db = self._get_expmeta_db(namespace)
        with self._write_lock, self._env.begin(write=True) as txn:
            return handler(
                StoreMutation(
                    store=self,
                    txn=txn,
                    db=db,
                    exp_db=exp_db,
                    expmeta_db=expmeta_db,
                )
            )

    def _txn_put(
        self,
        txn: Any,
        db: Any,
        exp_db: Any,
        expmeta_db: Any,
        key_bytes: bytes,
        value: object,
        expire_ts: int,
    ) -> None:
        value_bytes = self._encode_value(value)

        old_meta = txn.get(key_bytes, db=expmeta_db)
        if old_meta is not None:
            old_ts = _EXPIRY_STRUCT.unpack(old_meta[:8])[0]
            if old_ts:
                bucket = _BUCKET_STRUCT.pack(old_ts // 60)
                txn.delete(bucket, key_bytes, db=exp_db)
            txn.delete(key_bytes, db=expmeta_db)

        expiry_bytes = _EXPIRY_STRUCT.pack(expire_ts)
        txn.put(key_bytes, expiry_bytes + value_bytes, db=db)

        if expire_ts:
            bucket = _BUCKET_STRUCT.pack(expire_ts // 60)
            txn.put(bucket, key_bytes, db=exp_db, dupdata=True)
            txn.put(key_bytes, expiry_bytes, db=expmeta_db)

    def _txn_get(self, txn: Any, db: Any, key_bytes: bytes) -> object | None:
        raw = txn.get(key_bytes, db=db)
        return self._decode_raw_value(raw, time.time())

    def _txn_delete(
        self,
        txn: Any,
        db: Any,
        exp_db: Any,
        expmeta_db: Any,
        key_bytes: bytes,
    ) -> bool:
        meta = txn.get(key_bytes, db=expmeta_db)
        if meta:
            old_ts = _EXPIRY_STRUCT.unpack(meta[:8])[0]
            if old_ts:
                bucket = _BUCKET_STRUCT.pack(old_ts // 60)
                txn.delete(bucket, key_bytes, db=exp_db)
            txn.delete(key_bytes, db=expmeta_db)
        return txn.delete(key_bytes, db=db)

    @staticmethod
    def _decode_raw_value(raw: bytes | None, now: float) -> object | None:
        """Decode a raw LMDB payload, skipping expired entries."""
        if raw is None:
            return None
        expire_ts = _EXPIRY_STRUCT.unpack(raw[:8])[0]
        if expire_ts and expire_ts <= int(now):
            return None
        return cbor2.loads(raw[8:])

    @staticmethod
    def _position_cursor(
        *,
        cursor: Any,
        after_key: bytes | None,
        reverse: bool,
    ) -> bool:
        """Position a cursor on the first item of a page."""
        if after_key is None:
            return cursor.last() if reverse else cursor.first()

        if reverse:
            if not cursor.set_range(after_key):
                return cursor.last()
            while cursor.key() >= after_key:
                if not cursor.prev():
                    return False
            return True

        if not cursor.set_range(after_key):
            return False
        if cursor.key() == after_key:
            return cursor.next()
        return True

    def _cleanup_env(self, namespaces: list[str], now: float) -> None:
        now_bucket = int(now) // 60
        deleted_total = 0
        for ns in namespaces:
            if deleted_total >= self._cleanup_max_deletes:
                break
            try:
                db = self._get_db(ns)
                exp_db = self._get_exp_db(ns)
                expmeta_db = self._get_expmeta_db(ns)
                with self._write_lock, self._env.begin(write=True) as txn:
                    cursor = txn.cursor(db=exp_db)
                    to_delete: list[tuple[bytes, bytes]] = []
                    if cursor.first():
                        for bucket_bytes, key_bytes in cursor.iternext(keys=True, values=True):
                            bucket = _BUCKET_STRUCT.unpack(bucket_bytes)[0]
                            if bucket > now_bucket:
                                break
                            to_delete.append((bucket_bytes, key_bytes))
                            if len(to_delete) >= self._cleanup_max_deletes - deleted_total:
                                break
                    for bucket_bytes, key_bytes in to_delete:
                        raw = txn.get(key_bytes, db=db)
                        if raw is not None:
                            expire_ts = _EXPIRY_STRUCT.unpack(raw[:8])[0]
                            if expire_ts and expire_ts <= int(now):
                                txn.delete(key_bytes, db=db)
                                txn.delete(key_bytes, db=expmeta_db)
                                deleted_total += 1
                        txn.delete(bucket_bytes, key_bytes, db=exp_db)
            except Exception:
                logger.exception("Store cleanup failed for namespace: %s", ns)

    def _list_namespaces(self) -> list[str]:
        result: list[str] = []
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor(db=self._meta_db)
            if cursor.first():
                for key_bytes in cursor.iternext(keys=True, values=False):
                    if key_bytes.startswith(b"ns:"):
                        result.append(key_bytes[3:].decode("utf-8"))
        return result

    def _register_namespace_in_meta(self, namespace: str) -> None:
        meta_key = b"ns:" + namespace.encode("utf-8")
        with self._write_lock, self._env.begin(write=True) as txn:
            if txn.get(meta_key, db=self._meta_db) is None:
                txn.put(meta_key, b"1", db=self._meta_db)

    async def _register_namespace(self, namespace: str) -> None:
        async with self._namespace_lock:
            if namespace in self._namespaces:
                return
            self._get_db(namespace)
            self._get_exp_db(namespace)
            self._get_expmeta_db(namespace)
            self._register_namespace_in_meta(namespace)
            self._namespaces.add(namespace)

    # ------------------------------------------------------------------
    # DB cache helpers
    #
    # These use threading.Lock (not asyncio.Lock) deliberately:
    # open_db() is a synchronous LMDB call that completes in microseconds.
    # The lock is only contended on the first access to a new namespace.
    # ------------------------------------------------------------------

    def _get_db(self, namespace: str) -> Any:
        db = self._db_cache.get(namespace)
        if db is None:
            with self._db_cache_lock:
                db = self._db_cache.get(namespace)
                if db is None:
                    with self._write_lock:
                        db = self._env.open_db(namespace.encode("utf-8"), create=True)
                    self._db_cache[namespace] = db
        return db

    def _get_exp_db(self, namespace: str) -> Any:
        db = self._exp_cache.get(namespace)
        if db is None:
            with self._db_cache_lock:
                db = self._exp_cache.get(namespace)
                if db is None:
                    with self._write_lock:
                        db = self._env.open_db(
                            f"__exp__{namespace}".encode(),
                            create=True,
                            dupsort=True,
                        )
                    self._exp_cache[namespace] = db
        return db

    def _get_expmeta_db(self, namespace: str) -> Any:
        db = self._expmeta_cache.get(namespace)
        if db is None:
            with self._db_cache_lock:
                db = self._expmeta_cache.get(namespace)
                if db is None:
                    with self._write_lock:
                        db = self._env.open_db(
                            f"__expmeta__{namespace}".encode(),
                            create=True,
                        )
                    self._expmeta_cache[namespace] = db
        return db

    @asynccontextmanager
    async def _operation(self):
        async with self._operation_lock:
            self._assert_open()
            self._active_operations += 1
            self._operations_drained.clear()
        try:
            yield
        finally:
            async with self._operation_lock:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._operations_drained.set()

    @staticmethod
    def _esc_namespace(namespace: str) -> str:
        if any(namespace.startswith(p) for p in _RESERVED_PREFIXES):
            raise ValueError(f"Reserved namespace prefix: {namespace!r}")
        return namespace

    @staticmethod
    def _esc_key(key: str) -> bytes:
        return key.encode("utf-8")

    def _validate_key_size(self, key_bytes: bytes) -> None:
        if len(key_bytes) > self._max_key_bytes:
            raise ValueError(
                f"Store key exceeds limit: {len(key_bytes)} bytes > {self._max_key_bytes} max"
            )

    def _encode_value(self, value: object) -> bytes:
        value_bytes = cbor2.dumps(value)
        if len(value_bytes) > self._max_value_bytes:
            raise ValueError(
                "Store value exceeds limit: "
                f"{len(value_bytes)} bytes > {self._max_value_bytes} max"
            )
        return value_bytes

    def _assert_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("StoreService is closed")
