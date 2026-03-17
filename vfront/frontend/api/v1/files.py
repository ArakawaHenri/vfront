"""POST/GET/DELETE /v1/files — Files API endpoints.

Implements:
- POST /v1/files — upload (multipart/form-data)
- GET /v1/files — list (with purpose filter + pagination)
- GET /v1/files/{file_id} — retrieve metadata
- DELETE /v1/files/{file_id} — delete
- GET /v1/files/{file_id}/content — download binary content
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapiex.di import Inject
from starlette.responses import Response

from vfront.frontend.middleware.exceptions import InvalidRequestError, NotFoundError
from vfront.frontend.service.file_storage.main import FileStorageService
from vfront.frontend.service.jobs.store import JobStoreService
from vfront.protocol.file import FileDeleted, FileListResponse, FileObject

logger = logging.getLogger(__name__)

router = APIRouter()


async def _ensure_file_not_referenced_by_active_batch(
    file_id: str,
    job_store: JobStoreService,
) -> None:
    for execution in await job_store.list_batch_executions():
        if execution.status not in ("queued", "running"):
            continue
        spec = await job_store.get_batch_spec(execution.batch_id)
        if spec is None or spec.input_file_id != file_id:
            continue
        raise InvalidRequestError(
            f"Cannot delete file '{file_id}' because active batch "
            f"'{execution.batch_id}' still depends on it."
        )


@router.post("/files", response_model=FileObject)
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("user_data"),
    storage: FileStorageService = Inject("file_storage_service"),
):
    try:
        await file.seek(0)
        return await storage.upload_stream(
            filename=file.filename or "unknown",
            stream=file.file,
            purpose=purpose,
        )
    finally:
        await file.close()


@router.get("/files", response_model=FileListResponse)
async def list_files(
    purpose: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(10000, ge=1, le=10000),
    order: str = Query("desc"),
    storage: FileStorageService = Inject("file_storage_service"),
):
    files, has_more = await storage.list_files(
        purpose=purpose, after=after, limit=limit, order=order
    )
    return FileListResponse(
        data=files,
        has_more=has_more,
        first_id=files[0].id if files else None,
        last_id=files[-1].id if files else None,
    )


@router.get("/files/{file_id}", response_model=FileObject)
async def get_file(
    file_id: str,
    storage: FileStorageService = Inject("file_storage_service"),
):
    meta = await storage.get_metadata(file_id)
    if meta is None:
        raise NotFoundError(f"File '{file_id}' not found.")
    return meta


@router.delete("/files/{file_id}", response_model=FileDeleted)
async def delete_file(
    file_id: str,
    storage: FileStorageService = Inject("file_storage_service"),
    job_store: JobStoreService = Inject("job_store_service"),
):
    await _ensure_file_not_referenced_by_active_batch(file_id, job_store)
    result = await storage.delete(file_id)
    return result


@router.get("/files/{file_id}/content")
async def get_file_content(
    file_id: str,
    storage: FileStorageService = Inject("file_storage_service"),
):
    # Verify file exists
    meta = await storage.get_metadata(file_id)
    if meta is None:
        raise NotFoundError(f"File '{file_id}' not found.")

    content = await storage.get_content(file_id)
    if content is None:
        raise NotFoundError(f"Content for file '{file_id}' not found.")

    # RFC 5987 safe filename for Content-Disposition
    safe_name = meta.filename.replace("\\", "\\\\").replace('"', '\\"')
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
        },
    )
