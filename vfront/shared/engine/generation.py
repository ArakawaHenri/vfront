"""Shared helpers for consuming engine generation streams safely."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


async def collect_final_stream_output[TOutput](
    output_stream: AsyncIterator[TOutput],
    *,
    empty_error_message: str,
) -> TOutput:
    """Consume an async output stream, closing it on exit and returning the last item."""
    final_output: TOutput | None = None
    primary_error: BaseException | None = None
    try:
        async for output in output_stream:
            final_output = output
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if hasattr(output_stream, "aclose"):
            try:
                await output_stream.aclose()
            except asyncio.CancelledError:
                if primary_error is None:
                    raise
                logger.warning(
                    "Suppressed engine output stream cancellation during cleanup "
                    "while propagating the primary generation error.",
                    exc_info=True,
                )
            except Exception:
                if primary_error is None:
                    raise
                logger.exception(
                    "Failed to close engine output stream during cleanup; "
                    "preserving the primary generation error."
                )

    if final_output is None:
        raise RuntimeError(empty_error_message)
    return final_output
