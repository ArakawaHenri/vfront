"""Internal durable job specifications and execution state."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from vfront.protocol.response import ResponseCreateRequest

ExecutionStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class JobResolvedTarget(BaseModel):
    """Resolved routing snapshot captured when the job is created."""

    requested_model: str
    public_name: str
    backend: str
    runtime_model: str
    adapter: str | None = None
    tool_parser: str | None = None


class JobLease(BaseModel):
    """Distributed lease for a running background job."""

    object: Literal["job.lease"] = "job.lease"
    job_id: str
    owner_id: str
    lease_expires_at: float
    heartbeat_at: float = Field(default_factory=time.time)
    attempt: int = 0


class ResponseExecution(BaseModel):
    """Internal execution state for a background Responses job."""

    object: Literal["response.exec"] = "response.exec"
    response_id: str
    status: ExecutionStatus = "queued"
    requested_model: str
    target: JobResolvedTarget
    cancel_requested: bool = False
    attempt: int = 0
    owner_id: str | None = None
    last_error: str | None = None
    created_at: float
    updated_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


class BatchExecution(BaseModel):
    """Internal execution state for a background Batch job."""

    object: Literal["batch.exec"] = "batch.exec"
    batch_id: str
    status: ExecutionStatus = "queued"
    endpoint: str
    cancel_requested: bool = False
    attempt: int = 0
    owner_id: str | None = None
    last_error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


class ResponseJobSpec(BaseModel):
    """Persisted spec for a background Responses job."""

    object: Literal["response.job"] = "response.job"
    response_id: str
    created_at: float
    request: ResponseCreateRequest
    target: JobResolvedTarget
    conversation_messages: list[dict[str, Any]] = Field(default_factory=list)


class BatchJobSpec(BaseModel):
    """Persisted spec for a background Batch job."""

    object: Literal["batch.job"] = "batch.job"
    batch_id: str
    endpoint: str
    input_file_id: str
    completion_window: str = "24h"
    metadata: dict[str, str] | None = None
    request_targets: dict[str, JobResolvedTarget] = Field(default_factory=dict)
