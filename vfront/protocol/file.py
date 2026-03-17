"""OpenAI Files API protocol models.

Strictly aligned with the OpenAI Python SDK types:
- FileObject: metadata for an uploaded file
- FileDeleted: deletion confirmation
- FileListResponse: paginated file listing
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Purpose enum — all values from the SDK
# ---------------------------------------------------------------------------
FilePurpose = Literal[
    "assistants",
    "batch",
    "batch_output",
    "fine-tune",
    "fine-tune-results",
    "vision",
    "user_data",
    "evals",
    "assistants_output",
]


# ---------------------------------------------------------------------------
# FileObject — metadata returned for every file operation
# ---------------------------------------------------------------------------
class FileObject(BaseModel):
    id: str = Field(default_factory=lambda: f"file-{uuid.uuid4().hex[:24]}")
    object: Literal["file"] = "file"
    bytes: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    filename: str = ""
    purpose: str = "user_data"
    status: Literal["uploaded", "processed", "error"] = "uploaded"
    status_details: str | None = None
    expires_at: int | None = None


# ---------------------------------------------------------------------------
# FileDeleted — DELETE response
# ---------------------------------------------------------------------------
class FileDeleted(BaseModel):
    id: str
    object: Literal["file"] = "file"
    deleted: bool = True


# ---------------------------------------------------------------------------
# FileListResponse — paginated list (cursor-based)
# ---------------------------------------------------------------------------
class FileListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject] = []
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None
