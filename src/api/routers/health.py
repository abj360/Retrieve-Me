#!/usr/bin/env python3
"""
health.py --- liveness and readiness endpoints

Contains:
    healthz(): reports whether the process is alive
    readyz(): reports whether backing services are reachable
"""

import redis
from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient

from src.api.dependencies import get_settings

router = APIRouter()

PING_TIMEOUT_SECONDS = 1.5


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Reports liveness of the service process.

    Returns:
        status: Static ok payload while the process runs.
    """
    return {"status": "ok"}


@router.get("/readyz")
def readyz() -> dict[str, str]:
    """Reports readiness by pinging Qdrant and Redis.

    Returns:
        status: ready plus per-dependency check results.
    """
    settings = get_settings()
    checks = {"qdrant": "ok", "redis": "ok"}
    try:
        QdrantClient(url=settings.qdrant_url, timeout=PING_TIMEOUT_SECONDS).get_collections()
    except Exception:
        checks["qdrant"] = "unreachable"
    try:
        redis.from_url(settings.redis_url, socket_timeout=PING_TIMEOUT_SECONDS).ping()
    except Exception:
        checks["redis"] = "unreachable"
    degraded = {name for name, state in checks.items() if state != "ok"}
    if degraded:
        raise HTTPException(
            status_code=503,
            detail={"status": "not ready", "failed": sorted(degraded)},
        )
    return {"status": "ready", **checks}
