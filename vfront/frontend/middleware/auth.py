"""API key authentication middleware for OpenAI-compatible routes."""

from __future__ import annotations

import secrets

from fastapiex.settings import GetSettings
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from vfront.frontend.middleware.exceptions import AuthenticationError


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Enforce Bearer auth on `/v1/*` routes when `vfront.api_key` is configured."""

    @staticmethod
    def _expected_api_key() -> str:
        """Resolve the configured API key from fastapiex.settings."""
        app_settings = GetSettings("frontend.app")
        return (app_settings.api_key or "").strip()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not request.url.path.startswith("/v1"):
            return await call_next(request)

        expected_api_key = self._expected_api_key()
        if not expected_api_key:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        scheme, _, token = auth_header.partition(" ")

        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token,
            expected_api_key,
        ):
            exc = AuthenticationError()
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": exc.message,
                        "type": exc.error_type,
                        "param": exc.param,
                        "code": exc.code,
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
