"""Batch adapter — JSONL parsing, request dispatch, output assembly.

Handles:
- Parsing batch input JSONL files
- Dispatching individual requests to the appropriate endpoint handler
- Assembling output/error JSONL files
- Semaphore-limited concurrent execution
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from vfront.protocol.batch import (
    BatchRequestInput,
    BatchRequestOutput,
    BatchRequestOutputError,
    BatchRequestOutputResponse,
)

logger = logging.getLogger(__name__)


def parse_batch_input(content: bytes) -> list[BatchRequestInput]:
    """Parse a JSONL file into a list of BatchRequestInput objects.

    Raises ValueError with line number on parse errors.
    """
    lines = content.decode("utf-8").strip().split("\n")
    requests: list[BatchRequestInput] = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            req = BatchRequestInput.model_validate(data)
            requests.append(req)
        except Exception as e:
            raise ValueError(f"Line {i}: invalid batch request — {e}") from e
    return requests


def validate_batch_endpoint(endpoint: str) -> None:
    """Validate that the batch endpoint is supported."""
    valid = {"/v1/chat/completions", "/v1/embeddings", "/v1/completions", "/v1/responses"}
    if endpoint not in valid:
        raise ValueError(
            f"Unsupported batch endpoint: {endpoint}. Supported: {', '.join(sorted(valid))}"
        )


async def process_batch_requests(
    requests: list[BatchRequestInput],
    endpoint: str,
    dispatch_fn: Any,
    on_progress: Any = None,
    max_concurrency: int = 64,
) -> list[BatchRequestOutput]:
    """Process all batch requests concurrently.

    Args:
        requests: Parsed batch input lines.
        endpoint: The endpoint all requests target (e.g., "/v1/chat/completions").
        dispatch_fn: async callable(endpoint, body) -> dict (response body).
        on_progress: optional async callable(completed, failed) for progress updates.
        max_concurrency: Maximum number of concurrent vLLM requests.

    Returns:
        List of BatchRequestOutput objects.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _process_one(req: BatchRequestInput) -> BatchRequestOutput:
        async with sem:
            try:
                body = await dispatch_fn(endpoint, req.body)
                return BatchRequestOutput(
                    custom_id=req.custom_id,
                    response=BatchRequestOutputResponse(
                        status_code=200,
                        body=body,
                    ),
                    error=None,
                )
            except Exception as e:
                logger.warning("Batch request failed: custom_id=%s error=%s", req.custom_id, e)
                return BatchRequestOutput(
                    custom_id=req.custom_id,
                    response=BatchRequestOutputResponse(
                        status_code=getattr(e, "status_code", 500),
                        body={"error": {"message": str(e)}},
                    ),
                    error=BatchRequestOutputError(
                        code=getattr(e, "error_type", "server_error"),
                        message=str(e),
                    ),
                )

    # Execute all concurrently
    results = await asyncio.gather(*[_process_one(req) for req in requests])
    return list(results)


def build_output_jsonl(outputs: list[BatchRequestOutput]) -> bytes:
    """Serialize batch outputs to JSONL bytes."""
    lines = []
    for output in outputs:
        lines.append(output.model_dump_json())
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_error_jsonl(outputs: list[BatchRequestOutput]) -> bytes | None:
    """Build error JSONL containing only failed requests. Returns None if no errors."""
    errors = [o for o in outputs if o.error is not None]
    if not errors:
        return None
    lines = [o.model_dump_json() for o in errors]
    return ("\n".join(lines) + "\n").encode("utf-8")
