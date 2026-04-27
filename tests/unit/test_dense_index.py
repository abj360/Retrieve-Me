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


def test_upsert_empty_list_is_noop() -> None:
    """Asserts an empty upsert skips the client entirely."""
    client = FakeQdrantClient()
    assert make_index(client).upsert([], [], []) == 0
    assert client.upsert_calls == 0


class FakeQdrantClient:
    """Records upsert calls and answers collection queries in memory."""

    def __init__(self) -> None:
        """Creates an empty fake with one collection."""
        self.points = []
        self.upsert_calls = 0

    def collection_exists(self, collection: str) -> bool:
        """Returns whether the fake collection exists.

        Args:
            collection: Collection name to check.

        Returns:
            exists: Always True for the fake.
        """
        return True

    def upsert(self, collection_name: str, points: list, wait: bool = True) -> None:
        """Records one upsert call.

        Args:
            collection_name: Target collection.
            points: Points being written.
            wait: Whether to wait for indexing; ignored.
        """
        self.upsert_calls += 1
        self.points.extend(points)

    def count(self, collection: str):
        """Counts stored fake points.

        Args:
            collection: Collection name to count.

        Returns:
            result: Object with a count attribute.
        """
        return type("CountResult", (), {"count": len(self.points)})()

    def get_collections(self) -> list:
        """Returns a stub collections listing.

        Returns:
            collections: Empty list.
        """
        return []


class FakePool:
    """Yields one fake client with the pool acquire protocol."""

    def __init__(self, client: FakeQdrantClient) -> None:
        """Stores the fake client to yield.

        Args:
            client: Fake client yielded on every acquire.
        """
        self._client = client

    def acquire(self):
        """Yields the fake client as a context manager."""
        from contextlib import contextmanager

        @contextmanager
        def _acquire():
            yield self._client

        return _acquire()


def make_index(client: FakeQdrantClient):
    """Builds a DenseIndex over a fake client.

    Args:
        client: Fake client the index talks to.

    Returns:
        index: DenseIndex wired to the fake.
    """
    from src.ingest.dense_index import DenseIndex, QdrantConfig

    return DenseIndex(QdrantConfig(), FakePool(client))


def test_count_zero_when_collection_missing() -> None:
    """Asserts count() returns 0 instead of raising on a missing collection."""
    client = FakeQdrantClient()
    client.collection_exists = lambda collection: False
    assert make_index(client).count() == 0


def test_is_empty_on_fresh_collection() -> None:
    """Asserts is_empty is True before any upsert."""
    assert make_index(FakeQdrantClient()).is_empty() is True
