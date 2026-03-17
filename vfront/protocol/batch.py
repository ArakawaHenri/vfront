"""OpenAI Batch API protocol models.

Strictly aligned with the OpenAI Python SDK types:
- Batch: batch job metadata
- BatchRequestCounts: progress counters
- BatchError / BatchErrors: error details
- BatchRequestInput / BatchRequestOutput: JSONL line formats
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Status enum — all values from the SDK
# ---------------------------------------------------------------------------
BatchStatus = Literal[
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
]

# Supported batch endpoints
BatchEndpoint = Literal[
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/completions",
    "/v1/responses",
]


# ---------------------------------------------------------------------------
# Request counts
# ---------------------------------------------------------------------------
class BatchRequestCounts(BaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class BatchError(BaseModel):
    code: str | None = None
    message: str | None = None
    param: str | None = None
    line: int | None = None


class BatchErrors(BaseModel):
    object: Literal["list"] = "list"
    data: list[BatchError] = []


# ---------------------------------------------------------------------------
# Batch usage — aggregated token usage
# ---------------------------------------------------------------------------
class BatchUsageInputTokensDetails(BaseModel):
    cached_tokens: int = 0


class BatchUsageOutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class BatchUsage(BaseModel):
    input_tokens: int = 0
    input_tokens_details: BatchUsageInputTokensDetails = Field(
        default_factory=BatchUsageInputTokensDetails
    )
    output_tokens: int = 0
    output_tokens_details: BatchUsageOutputTokensDetails = Field(
        default_factory=BatchUsageOutputTokensDetails
    )
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Batch object — main metadata model
# ---------------------------------------------------------------------------
class Batch(BaseModel):
    id: str = Field(default_factory=lambda: f"batch_{uuid.uuid4().hex[:24]}")
    object: Literal["batch"] = "batch"
    endpoint: str = ""
    errors: BatchErrors | None = None
    input_file_id: str = ""
    completion_window: str = "24h"
    status: BatchStatus = "validating"
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: BatchRequestCounts = Field(default_factory=BatchRequestCounts)
    metadata: dict[str, str] | None = None
    model: str | None = None
    usage: BatchUsage | None = None


# ---------------------------------------------------------------------------
# Batch list response (cursor-based pagination)
# ---------------------------------------------------------------------------
class BatchListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[Batch] = []
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


# ---------------------------------------------------------------------------
# JSONL line formats — for batch input/output files
# ---------------------------------------------------------------------------
class BatchRequestInput(BaseModel):
    """A single line in the batch input JSONL file."""

    custom_id: str
    method: Literal["POST"] = "POST"
    url: str
    body: dict[str, Any]


class BatchRequestOutputResponse(BaseModel):
    """The response portion of a batch output line."""

    status_code: int = 200
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")
    body: dict[str, Any] = {}


class BatchRequestOutputError(BaseModel):
    """Error portion of a batch output line."""

    code: str
    message: str


class BatchRequestOutput(BaseModel):
    """A single line in the batch output/error JSONL file."""

    id: str = Field(default_factory=lambda: f"batch_req_{uuid.uuid4().hex[:24]}")
    custom_id: str = ""
    response: BatchRequestOutputResponse | None = None
    error: BatchRequestOutputError | None = None
