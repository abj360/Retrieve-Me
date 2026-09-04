#!/usr/bin/env python3
"""
main.py --- FastAPI application entrypoint for the retrieval service

Contains:
    configure_logging(): applies RETRIEVAL_LOG_LEVEL to the service's loggers
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
from src.api.routers import documents, health, retrieve

logger = logging.getLogger("retrieval.startup")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str) -> None:
    """Applies the configured log level to the service's own loggers.

    uvicorn configures its own loggers and leaves the root logger alone, so
    without this every logger under src/ sits at WARNING and the ingest,
    index-load and warmup diagnostics never reach the log.

    Args:
        level: Level name from RETRIEVAL_LOG_LEVEL, e.g. "INFO".
    """
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        logging.basicConfig(format=LOG_FORMAT)
        logger.warning("unknown RETRIEVAL_LOG_LEVEL %r, leaving levels untouched", level)
        return
    logging.basicConfig(level=resolved, format=LOG_FORMAT)
    logging.getLogger("retrieval").setLevel(resolved)
    logging.getLogger("src").setLevel(resolved)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Stores shared settings and warms the models for the app's lifetime.

    Args:
        app: Application instance the lifespan is bound to.
    """
    settings = get_settings()
    app.state.settings = settings
    logger.info("Retrieve-Me %s starting at log level %s", settings.app_version, settings.log_level)
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
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_title, version=settings.app_version, lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(retrieve.router)
    app.include_router(documents.router)
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
