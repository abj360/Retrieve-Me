#!/usr/bin/env python3
"""
health.py --- liveness and readiness endpoints

Contains:
    healthz(): reports whether the process is alive, plus service version
    check_qdrant(): returns ok when Qdrant answers within the ping timeout
    check_redis(): returns ok when Redis answers within the ping timeout
    readyz(): reports whether backing services are reachable
"""

import os

import redis
from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient

from src.api.dependencies import Settings, get_settings

# readiness probes stay cheap: no index loads, no model calls

router = APIRouter()

PING_TIMEOUT_SECONDS = 1.5


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Reports liveness of the service process.

    Returns:
        status: Static ok payload while the process runs.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_title,
        "pid": os.getpid(),
        "version": settings.app_version,
    }


def check_qdrant(settings: Settings) -> str:
    """Returns ok when Qdrant answers within the ping timeout.

    Args:
        settings: Service settings carrying the Qdrant URL.

    Returns:
        state: "ok" or "unreachable".
    """
    try:
        QdrantClient(url=settings.qdrant_url, timeout=PING_TIMEOUT_SECONDS).get_collections()
    except Exception:
        return "down"
    return "ok"


def check_redis(settings: Settings) -> str:
    """Returns ok when Redis answers within the ping timeout.

    Args:
        settings: Service settings carrying the Redis URL.

    Returns:
        state: "ok" or "unreachable".
    """
    try:
        redis.from_url(settings.redis_url, socket_timeout=PING_TIMEOUT_SECONDS).ping()
    except Exception:
        return "unreachable"
    return "ok"


@router.get("/readyz")
def readyz() -> dict[str, str]:
    """Reports readiness by pinging Qdrant and Redis within the ping timeout.

    Returns:
        status: ready plus per-dependency check results.
    """
    settings = get_settings()
    checks = {"qdrant": check_qdrant(settings), "redis": check_redis(settings)}
    degraded = {name for name, state in checks.items() if state != "ok"}
    if degraded:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "failed": sorted(degraded)},
        )
    return {"status": "ready", **checks}
