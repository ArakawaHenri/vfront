"""File storage service — filesystem content + LMDB metadata.

Adapted from neo_tx_classifier TempFileService pattern:
- File content stored on filesystem (supports large files)
- File metadata (FileObject) stored in LMDB via StoreService
- TTL-based cleanup with periodic background scanning
- All blocking I/O wrapped in asyncio.to_thread()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

from fastapiex.di import BaseService, Require, Service
from fastapiex.settings import GetSettings

from vfront.frontend.service.store.lmdb import StoreService
from vfront.protocol.file import FileDeleted, FileObject

logger = logging.getLogger(__name__)

_NAMESPACE = "files"
_LOCK_FILE = ".file_cleanup.lock"
# Characters not safe for filenames
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*%\x00-\x1f]')
# file_id must match our generated pattern — reject anything else
_FILE_ID_RE = re.compile(r"^file-[a-f0-9]{24}$")
_CREATED_AT_INDEX_NAMESPACE = "files_idx_created_at"

# Restrict file extensions for purposes that only accept specific formats.
# Purposes not listed here accept any file type.
_PURPOSE_ALLOWED_SUFFIXES: dict[str, frozenset[str]] = {
    "batch": frozenset({".jsonl"}),
    "fine-tune": frozenset({".jsonl"}),
    "vision": frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"}),
}


def _sanitize_filename(name: str) -> str:
    """Replace unsafe characters with URL-encoded equivalents."""
    return _UNSAFE_RE.sub(lambda m: f"%{ord(m.group()):02X}", name)


def _validate_file_purpose_extension(filename: str, purpose: str) -> None:
    """Raise InvalidRequestError if the file extension doesn't match the purpose."""
    allowed = _PURPOSE_ALLOWED_SUFFIXES.get(purpose)
    if allowed is None:
        return
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        from vfront.frontend.middleware.exceptions import InvalidRequestError
        raise InvalidRequestError(
            f"Files with purpose '{purpose}' must use one of these extensions: "
            f"{', '.join(sorted(allowed))}. Got: '{suffix or '(none)'}'."
        )


def _validate_file_id(file_id: str) -> None:
    """Reject file IDs that don't match our generated pattern (path traversal guard)."""
    if not _FILE_ID_RE.match(file_id):
        from vfront.frontend.middleware.exceptions import InvalidRequestError
        raise InvalidRequestError(f"Invalid file ID format: '{file_id}'.")


def _created_at_index_key(created_at: int, object_id: str) -> str:
    return f"{created_at:020d}:{object_id}"


def _purpose_index_namespace(purpose: str) -> str:
    return f"files_idx_purpose_{quote(purpose, safe='')}"


async def _best_effort_unlink(path: Path, *, warning_message: str) -> bool:
    try:
        await asyncio.to_thread(path.unlink, True)
        return True
    except Exception:
        logger.warning(warning_message, path, exc_info=True)
        return False


@Service("file_storage_service", eager=True)
class FileStorageService(BaseService):
    """Filesystem-backed file storage with LMDB metadata tracking."""

    _base_dir: Path
    _max_file_size: int
    _retention_seconds: int
    _cleanup_interval: int
    _store: StoreService
    _cleanup_task: asyncio.Task | None
    _closed: bool

    @classmethod
    async def create(cls, store: StoreService = Require("store_service")) -> FileStorageService:  # type: ignore[assignment]
        cfg = GetSettings("frontend.file_storage")
        instance = cls()
        instance._base_dir = Path(cfg.base_dir)
        instance._max_file_size = cfg.max_file_size_mb * 1024 * 1024
        instance._retention_seconds = cfg.retention_days * 86400
        instance._cleanup_interval = cfg.cleanup_interval_seconds
        instance._cleanup_task = None
        instance._closed = False
        instance._store = store

        # Startup should avoid a thread-hop for a trivial local mkdir. Under the
        # ASGI lifespan test harness this extra hop can delay startup enough to
        # trip the default 5s timeout.
        instance._base_dir.mkdir(parents=True, exist_ok=True)

        # Start cleanup loop
        instance._cleanup_task = asyncio.create_task(instance._cleanup_loop())
        logger.info("FileStorageService started: base_dir=%s", instance._base_dir)
        return instance

    def _get_store(self) -> StoreService:
        """Return the injected StoreService."""
        return self._store

    def _file_path(self, file_id: str) -> Path:
        """Get the filesystem path for a file by ID."""
        return self._base_dir / file_id

    async def upload(
        self,
        filename: str,
        content: bytes,
        purpose: str = "user_data",
    ) -> FileObject:
        """Upload a file and store metadata."""
        _validate_file_purpose_extension(filename, purpose)
        if len(content) > self._max_file_size:
            from vfront.frontend.middleware.exceptions import InvalidRequestError
            raise InvalidRequestError(
                f"File size {len(content)} exceeds maximum {self._max_file_size} bytes."
            )

        file_id = f"file-{uuid.uuid4().hex[:24]}"
        now = int(time.time())
        expires_at = now + self._retention_seconds if self._retention_seconds > 0 else None

        # Write content to filesystem
        file_path = self._file_path(file_id)
        await asyncio.to_thread(self._write_file, file_path, content)

        file_obj = self._build_file_object(
            file_id=file_id,
            filename=filename,
            size_bytes=len(content),
            purpose=purpose,
            created_at=now,
            expires_at=expires_at,
        )
        await self._store_metadata(file_obj)

        logger.info("File uploaded: id=%s filename=%s bytes=%d purpose=%s",
                     file_id, filename, len(content), purpose)
        return file_obj

    async def upload_stream(
        self,
        filename: str,
        stream: BinaryIO,
        purpose: str = "user_data",
    ) -> FileObject:
        """Upload a file from a blocking file-like stream without buffering it all in memory."""
        _validate_file_purpose_extension(filename, purpose)
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        now = int(time.time())
        expires_at = now + self._retention_seconds if self._retention_seconds > 0 else None
        file_path = self._file_path(file_id)

        try:
            size_bytes = await asyncio.to_thread(
                self._write_stream,
                file_path,
                stream,
                self._max_file_size,
            )
        except ValueError as exc:
            await _best_effort_unlink(
                file_path,
                warning_message="Failed to clean up partially written upload file: %s",
            )
            from vfront.frontend.middleware.exceptions import InvalidRequestError
            raise InvalidRequestError(str(exc)) from exc
        except Exception:
            await _best_effort_unlink(
                file_path,
                warning_message="Failed to clean up partially written upload file: %s",
            )
            raise

        file_obj = self._build_file_object(
            file_id=file_id,
            filename=filename,
            size_bytes=size_bytes,
            purpose=purpose,
            created_at=now,
            expires_at=expires_at,
        )
        await self._store_metadata(file_obj)

        logger.info(
            "File uploaded: id=%s filename=%s bytes=%d purpose=%s",
            file_id,
            filename,
            size_bytes,
            purpose,
        )
        return file_obj

    async def get_metadata(self, file_id: str) -> FileObject | None:
        """Get file metadata by ID."""
        _validate_file_id(file_id)
        store = self._get_store()
        data = await store.get(namespace=_NAMESPACE, key=file_id)
        if data is None:
            return None
        return FileObject.model_validate(data)

    async def get_content(self, file_id: str) -> bytes | None:
        """Get file content by ID."""
        _validate_file_id(file_id)
        file_path = self._file_path(file_id)
        try:
            return await asyncio.to_thread(self._read_file, file_path)
        except FileNotFoundError:
            return None

    async def list_files(
        self,
        purpose: str | None = None,
        after: str | None = None,
        limit: int = 10000,
        order: str = "desc",
    ) -> tuple[list[FileObject], bool]:
        """List files with optional purpose filter and cursor pagination.

        Returns (files, has_more).
        """
        store = self._get_store()
        reverse = order == "desc"
        index_namespace = (
            _purpose_index_namespace(purpose) if purpose else _CREATED_AT_INDEX_NAMESPACE
        )

        await self._ensure_indices()
        after_index_key = await self._resolve_after_index_key(
            after=after,
            purpose=purpose,
        )
        if after and after_index_key is None:
            return [], False

        page_limit = max(limit, 1)
        files: list[FileObject] = []
        scan_after = after_index_key
        batch_size = min(max(page_limit * 2, 50), 500)

        while len(files) <= page_limit:
            page, has_more = await store.list_page(
                namespace=index_namespace,
                after=scan_after,
                limit=batch_size,
                reverse=reverse,
            )
            if not page:
                return files[:page_limit], False

            for _index_key, file_id in page:
                file_obj = await self.get_metadata(str(file_id))
                if file_obj is None:
                    continue
                files.append(file_obj)
                if len(files) > page_limit:
                    return files[:page_limit], True

            if not has_more:
                return files[:page_limit], False
            scan_after = page[-1][0]

        return files[:page_limit], len(files) > page_limit

    async def delete(self, file_id: str) -> FileDeleted:
        """Delete a file (content + metadata)."""
        _validate_file_id(file_id)
        store = self._get_store()

        existing = await self.get_metadata(file_id)
        if existing is not None:
            index_key = _created_at_index_key(existing.created_at, existing.id)
            await store.delete(namespace=_CREATED_AT_INDEX_NAMESPACE, key=index_key)
            await store.delete(
                namespace=_purpose_index_namespace(existing.purpose),
                key=index_key,
            )

        # Remove metadata
        existed = await store.delete(namespace=_NAMESPACE, key=file_id)

        # Remove content file
        file_path = self._file_path(file_id)
        await _best_effort_unlink(
            file_path,
            warning_message="Failed to remove deleted file content: %s",
        )

        return FileDeleted(id=file_id, deleted=bool(existed))

    # --- Filesystem helpers (blocking, run via to_thread) ---

    @staticmethod
    def _write_file(path: Path, content: bytes) -> None:
        path.write_bytes(content)

    @staticmethod
    def _write_stream(path: Path, stream: BinaryIO, max_file_size: int) -> int:
        total_bytes = 0
        with path.open("wb") as handle:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_file_size:
                    raise ValueError(
                        f"File size {total_bytes} exceeds maximum {max_file_size} bytes."
                    )
                handle.write(chunk)
        return total_bytes

    @staticmethod
    def _read_file(path: Path) -> bytes:
        return path.read_bytes()

    def _build_file_object(
        self,
        *,
        file_id: str,
        filename: str,
        size_bytes: int,
        purpose: str,
        created_at: int,
        expires_at: int | None,
    ) -> FileObject:
        return FileObject(
            id=file_id,
            bytes=size_bytes,
            created_at=created_at,
            filename=_sanitize_filename(filename),
            purpose=purpose,
            status="processed",
            expires_at=expires_at,
        )

    async def _store_metadata(self, file_obj: FileObject) -> None:
        store = self._get_store()
        retention_minutes = (
            self._retention_seconds // 60 if self._retention_seconds > 0 else None
        )
        await store.set(
            namespace=_NAMESPACE,
            key=file_obj.id,
            value=file_obj.model_dump(),
            retention=retention_minutes,
        )
        index_key = _created_at_index_key(file_obj.created_at, file_obj.id)
        await store.set(
            namespace=_CREATED_AT_INDEX_NAMESPACE,
            key=index_key,
            value=file_obj.id,
            retention=retention_minutes,
        )
        await store.set(
            namespace=_purpose_index_namespace(file_obj.purpose),
            key=index_key,
            value=file_obj.id,
            retention=retention_minutes,
        )

    async def _ensure_indices(self) -> None:
        """Backfill indices for files created before index namespaces existed."""
        store = self._get_store()
        existing_index, _ = await store.list_page(
            namespace=_CREATED_AT_INDEX_NAMESPACE,
            limit=1,
            reverse=True,
        )
        if existing_index:
            return

        file_keys = await store.list_keys(namespace=_NAMESPACE)
        if not file_keys:
            return

        for file_id in file_keys:
            data = await store.get(namespace=_NAMESPACE, key=file_id)
            if data is None:
                continue
            await self._store_metadata(FileObject.model_validate(data))

    async def _resolve_after_index_key(
        self,
        *,
        after: str | None,
        purpose: str | None,
    ) -> str | None:
        if after is None:
            return None

        after_obj = await self.get_metadata(after)
        if after_obj is None:
            return None
        if purpose is not None and after_obj.purpose != purpose:
            return None
        return _created_at_index_key(after_obj.created_at, after_obj.id)

    # --- Cleanup ---

    async def _cleanup_loop(self) -> None:
        """Periodically remove expired files."""
        while not self._closed:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("File cleanup error")

    async def _cleanup_expired(self) -> None:
        """Remove files whose metadata has expired from the store."""
        try:
            entries = await asyncio.to_thread(os.listdir, self._base_dir)
        except FileNotFoundError:
            return

        removed = 0
        for entry in entries:
            if entry.startswith("."):
                continue
            if not entry.startswith("file-"):
                continue
            # Check if metadata still exists
            store = self._get_store()
            data = await store.get(namespace=_NAMESPACE, key=entry)
            if data is None:
                # Metadata expired — remove orphan content file
                path = self._base_dir / entry
                if await _best_effort_unlink(
                    path,
                    warning_message="Failed to remove orphan file during cleanup: %s",
                ):
                    removed += 1

        if removed:
            logger.info("File cleanup: removed %d orphan files", removed)

    @classmethod
    async def destroy(cls, instance: FileStorageService) -> None:
        instance._closed = True
        if instance._cleanup_task and not instance._cleanup_task.done():
            instance._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await instance._cleanup_task
        logger.info("FileStorageService stopped")
