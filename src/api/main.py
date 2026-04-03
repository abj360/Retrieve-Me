#!/usr/bin/env python3
"""
main.py --- FastAPI application entrypoint for the retrieval service

Contains:
    create_app(): builds and configures the FastAPI application
"""

from fastapi import FastAPI

from src.api.middleware import RequestLoggingMiddleware
from src.api.routers import retrieve


def create_app() -> FastAPI:
    """Builds the FastAPI app and registers all routers.

    Returns:
        app: Configured FastAPI application instance.
    """
    app = FastAPI(title="retrieval-core")
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(retrieve.router)
    return app


app = create_app()
