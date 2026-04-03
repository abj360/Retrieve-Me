#!/usr/bin/env python3
"""
middleware.py --- HTTP middleware for request/response logging

Contains:
    RequestLoggingMiddleware: logs method, path, status, and duration per request
"""

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("retrieval.request")


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
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
