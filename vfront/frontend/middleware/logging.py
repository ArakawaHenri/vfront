"""Request logging middleware with X-Request-ID propagation."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse
from starlette.responses import StreamingResponse

from vfront.core.logging import request_id_ctx

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request and propagate X-Request-ID via contextvars."""

    @staticmethod
    async def _wrap_stream(
        body_iterator: AsyncIterable[str | bytes | memoryview],
        *,
        method: str,
        path: str,
        status_code: int,
        started_at: float,
        request_id: str,
    ) -> AsyncIterator[str | bytes | memoryview]:
        token = request_id_ctx.set(request_id)
        try:
            async for chunk in body_iterator:
                yield chunk
        except Exception:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "Request failed method=%s path=%s status=%s duration_ms=%.2f",
                method,
                path,
                status_code,
                duration_ms,
            )
            raise
        except BaseException:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "Request cancelled method=%s path=%s status=%s duration_ms=%.2f",
                method,
                path,
                status_code,
                duration_ms,
            )
            raise
        else:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Request completed method=%s path=%s status=%s duration_ms=%.2f",
                method,
                path,
                status_code,
                duration_ms,
            )
        finally:
            request_id_ctx.reset(token)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        started_at = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)

        client_host = request.client.host if request.client else "-"
        logger.info(
            "Request started method=%s path=%s client=%s",
            request.method,
            request.url.path,
            client_host,
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "Request failed method=%s path=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            if isinstance(response, StreamingResponse):
                response.body_iterator = self._wrap_stream(
                    response.body_iterator,
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    started_at=started_at,
                    request_id=request_id,
                )
                return response

            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Request completed method=%s path=%s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response
        finally:
            request_id_ctx.reset(token)
