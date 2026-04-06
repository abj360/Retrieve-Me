#!/usr/bin/env python3
"""
health.py --- liveness and readiness endpoints

Contains:
    healthz(): reports whether the process is alive
    readyz(): reports whether backing services are reachable
"""

import redis
from fastapi import APIRouter
from qdrant_client import QdrantClient

router = APIRouter()

PING_TIMEOUT_SECONDS = 2.0


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
    checks = {"qdrant": "ok", "redis": "ok"}
    try:
        QdrantClient(url="http://localhost:6333", timeout=PING_TIMEOUT_SECONDS).get_collections()
    except Exception:
        checks["qdrant"] = "unreachable"
    try:
        redis.from_url("redis://localhost:6379", socket_timeout=PING_TIMEOUT_SECONDS).ping()
    except Exception:
        checks["redis"] = "unreachable"
    return {"status": "ready", **checks}
