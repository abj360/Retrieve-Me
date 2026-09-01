#!/usr/bin/env python3
"""
main.py --- FastAPI application entrypoint for the retrieval service

Contains:
    lifespan(): stores shared settings and warms the models at startup
    create_app(): builds and configures the FastAPI application
    root(): returns basic service metadata
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_pipeline, get_settings
from src.api.middleware import RequestLoggingMiddleware
from src.api.routers import health, retrieve

logger = logging.getLogger("retrieval.startup")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Stores shared settings and warms the models for the app's lifetime.

    Args:
        app: Application instance the lifespan is bound to.
    """
    settings = get_settings()
    app.state.settings = settings
    if settings.warmup_on_startup:
        logger.info("warming retrieval models before accepting traffic")
        get_pipeline().warmup()
    yield


def create_app() -> FastAPI:
    """Builds the FastAPI app and registers all routers.

    Returns:
        app: Configured FastAPI application instance.
    """
    settings = get_settings()
    app = FastAPI(title=settings.app_title, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(retrieve.router)
    app.include_router(health.router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Returns basic service metadata.

        Returns:
            metadata: Service name and documentation pointer.
        """
        return {
            "service": settings.app_title,
            "version": settings.app_version,
            "docs": "/docs",
        }

    return app


app = create_app()
