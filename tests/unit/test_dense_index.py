#!/usr/bin/env python3
"""
test_dense_index.py --- unit tests for the Qdrant client pool and dense index

Contains:
    test_pool_is_bounded_under_concurrency(): asserts excess callers fail fast
    test_pool_hands_back_same_client(): asserts clients are reused round-robin
    test_retry_with_backoff_recovers_flaky_call(): asserts transient retries work
    test_retry_with_backoff_gives_up(): asserts exhaustion re-raises
"""

import time

import pytest

from src.ingest.dense_index import (
    PoolExhaustedError,
    QdrantClientPool,
    QdrantConfig,
    retry_with_backoff,
)


def make_pool(max_size: int = 2, acquire_timeout: float = 0.01) -> QdrantClientPool:
    """Builds a tiny pool with a near-zero acquire timeout.

    Args:
        max_size: Maximum clients the pool hands out.
        acquire_timeout: Seconds a caller waits before PoolExhaustedError.

    Returns:
        pool: Bounded pool of real (unconnected) Qdrant clients.
    """
    return QdrantClientPool(QdrantConfig(), max_size=max_size, acquire_timeout=acquire_timeout)


def test_pool_is_bounded_under_concurrency() -> None:
    """Asserts a third concurrent checkout fails fast instead of opening more clients."""
    pool = make_pool(max_size=2)
    with pool.acquire():
        with pool.acquire():
            with pytest.raises(PoolExhaustedError):
                with pool.acquire():
                    pass


def test_pool_hands_back_same_client() -> None:
    """Asserts clients are returned to the pool and reused."""
    pool = make_pool(max_size=1)
    with pool.acquire() as first:
        pass
    with pool.acquire() as second:
        pass
    assert first is second


def test_retry_with_backoff_recovers_flaky_call(monkeypatch) -> None:
    """Asserts a flaky operation succeeds within the retry budget."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    attempts = {"count": 0}

    def flaky() -> str:
        """Fails twice, then succeeds."""
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("simulated blip")
        return "ok"

    assert retry_with_backoff(flaky, retries=3) == "ok"
    assert attempts["count"] == 3


def test_retry_with_backoff_gives_up(monkeypatch) -> None:
    """Asserts exhaustion re-raises the last error."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    def always_fails() -> str:
        """Fails on every attempt."""
        raise TimeoutError("simulated outage")

    with pytest.raises(TimeoutError):
        retry_with_backoff(always_fails, retries=2)
