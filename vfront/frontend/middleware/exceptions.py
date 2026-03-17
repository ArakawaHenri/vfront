"""OpenAI-format exception hierarchy and handlers.

Every error response strictly follows:
{
    "error": {
        "message": "...",
        "type": "...",
        "param": null,
        "code": null
    }
}
"""

from __future__ import annotations

import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from vfront.shared.engine.errors import EngineInputError

logger = logging.getLogger(__name__)


# ── Exception hierarchy ──


class OpenAIError(Exception):
    """Base for all errors returned in OpenAI's JSON format."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "server_error"

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message
        self.param = param
        self.code = code
        super().__init__(message)


class InvalidRequestError(OpenAIError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "invalid_request_error"


class AuthenticationError(OpenAIError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_type = "authentication_error"

    def __init__(self, message: str = "Incorrect API key provided.", **kwargs: str | None) -> None:
        super().__init__(message, code="invalid_api_key", **kwargs)


class NotFoundError(OpenAIError):
    status_code = status.HTTP_404_NOT_FOUND
    error_type = "not_found_error"


class RateLimitError(OpenAIError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_type = "rate_limit_error"


class ServerError(OpenAIError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type = "server_error"


class OverloadedError(OpenAIError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "overloaded_error"


# ── Response builder ──


def _error_json(
    status_code: int,
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


# ── Exception handlers ──


async def openai_error_handler(request: Request, exc: OpenAIError) -> JSONResponse:
    logger.warning(
        "OpenAI error method=%s path=%s type=%s message=%s",
        request.method,
        request.url.path,
        exc.error_type,
        exc.message,
    )
    return _error_json(exc.status_code, exc.message, exc.error_type, exc.param, exc.code)


async def engine_input_error_handler(request: Request, exc: EngineInputError) -> JSONResponse:
    logger.warning(
        "Engine input error method=%s path=%s message=%s",
        request.method,
        request.url.path,
        exc.message,
    )
    return _error_json(
        status.HTTP_400_BAD_REQUEST,
        exc.message,
        "invalid_request_error",
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = first.get("loc", [])
        param = ".".join(str(part) for part in loc if part != "body")
        message = first.get("msg", str(exc))
    else:
        param = None
        message = str(exc)

    logger.warning(
        "Validation error method=%s path=%s errors=%s",
        request.method,
        request.url.path,
        errors,
    )
    return _error_json(
        status.HTTP_400_BAD_REQUEST,
        message,
        "invalid_request_error",
        param=param or None,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    type_mapping = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        405: "invalid_request_error",
        429: "rate_limit_error",
        503: "overloaded_error",
    }
    logger.info(
        "HTTP exception method=%s path=%s status=%s",
        request.method,
        request.url.path,
        exc.status_code,
    )
    return _error_json(
        exc.status_code,
        str(exc.detail or "HTTP error"),
        type_mapping.get(exc.status_code, "server_error"),
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception method=%s path=%s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return _error_json(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "An internal server error occurred.",
        "server_error",
    )
