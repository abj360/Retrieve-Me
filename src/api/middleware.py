#!/usr/bin/env python3
"""
middleware.py --- HTTP middleware for request/response logging

Contains:
    RequestLoggingMiddleware: logs method, path, status, and duration per request
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("retrieval.http")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs one line per HTTP request with status code and duration."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Logs the request and its response status with elapsed time.

        Args:
            request: Incoming HTTP request.
            call_next: Next handler in the middleware chain.

        Returns:
            response: Response produced by the downstream handler.
        """
        started = time.perf_counter()
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "%s %s %d %.1fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response
