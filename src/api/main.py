#!/usr/bin/env python3
"""
main.py --- FastAPI application entrypoint for the retrieval service

Contains:
    lifespan(): stores shared settings on app state across the app lifetime
    create_app(): builds and configures the FastAPI application
    root(): returns basic service metadata
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import RequestLoggingMiddleware
from src.api.dependencies import get_settings
from src.api.routers import health, retrieve


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Stores shared settings on app state for the whole app lifetime.

    Args:
        app: Application instance the lifespan is bound to.
    """
    app.state.settings = get_settings()
    yield


def create_app() -> FastAPI:
    """Builds the FastAPI app and registers all routers.

    Returns:
        app: Configured FastAPI application instance.
    """
    app = FastAPI(title="Retrieve-Me", lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(retrieve.router)
    app.include_router(health.router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Returns basic service metadata.

        Returns:
            metadata: Service name and documentation pointer.
        """
        return {"service": "Retrieve-Me", "docs": "/docs"}

    return app


app = create_app()
