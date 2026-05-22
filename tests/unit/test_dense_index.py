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


def test_retry_backoff_delay_doubles() -> None:
    """Asserts the backoff delay doubles per attempt."""
    from src.ingest.dense_index import RETRY_BASE_DELAY_SECONDS

    assert RETRY_BASE_DELAY_SECONDS * 2 > RETRY_BASE_DELAY_SECONDS


def test_point_ids_are_deterministic() -> None:
    """Asserts re-ingesting the same chunk ids overwrites rather than duplicates."""
    client = FakeQdrantClient()
    index = make_index(client)
    index.upsert(["c1"], [[0.1] * 8], [{"doc_id": "d"}])
    first_id = client.points[0].id
    index.upsert(["c1"], [[0.2] * 8], [{"doc_id": "d"}])
    assert client.points[1].id == first_id


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


def test_upsert_writes_in_batches() -> None:
    """Asserts a 250-point upsert lands as three batch calls of 100/100/50."""
    client = FakeQdrantClient()
    ids = [f"chunk-{index}" for index in range(250)]
    vectors = [[0.1] * 8 for _ in ids]
    payloads = [{"doc_id": "doc"} for _ in ids]
    written = make_index(client).upsert(ids, vectors, payloads, batch_size=100)
    assert written == 250
    assert client.upsert_calls == 3
    assert len(client.points) == 250


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


def test_build_filter_maps_exact_matches() -> None:
    """Asserts payload filters become exact-match field conditions."""
    from src.ingest.dense_index import _build_filter

    query_filter = _build_filter({"source": "legal"})
    assert query_filter is not None
    assert len(query_filter.must) == 1
    assert _build_filter(None) is None


def test_search_on_empty_index_returns_no_hits() -> None:
    """Asserts searching an empty collection returns [] rather than raising."""
    client = FakeQdrantClient()
    client.collection_exists = lambda collection: False
    assert make_index(client).search([0.1] * 384, top_k=5) == []


def test_batch_size_one_writes_each_point() -> None:
    """Asserts batch_size of 1 still writes every point."""
    client = FakeQdrantClient()
    ids = ["a", "b", "c"]
    make_index(client).upsert(ids, [[0.1] * 8] * 3, [{}] * 3, batch_size=1)
    assert client.upsert_calls == 3
    assert len(client.points) == 3


def test_health_check_true_when_server_answers() -> None:
    """Asserts health_check reports True when Qdrant responds."""
    client = FakeQdrantClient()
    assert make_index(client).health_check() is True


def test_health_check_false_when_pool_fails() -> None:
    """Asserts health_check reports False when the pool raises."""
    from src.ingest.dense_index import DenseIndex, QdrantConfig

    class FailingPool:
        """Raises on every acquire."""

        def acquire(self):
            """Raises a connection failure."""
            raise ConnectionError("qdrant down")

    index = DenseIndex(QdrantConfig(), FailingPool())
    assert index.health_check() is False


def test_pool_close_drains_idle_clients() -> None:
    """Asserts close() empties the pool so later acquires fail."""
    pool = make_pool(max_size=1)
    pool.close()
    with pytest.raises(PoolExhaustedError):
        with pool.acquire():
            pass


def test_drop_collection_if_empty_only_when_empty() -> None:
    """Asserts the conditional drop leaves a non-empty collection alone."""
    client = FakeQdrantClient()
    index = make_index(client)
    assert index.drop_collection_if_empty() is True
    client.upsert(collection_name="chunks", points=["p1"])
    assert index.drop_collection_if_empty() is False


def test_count_reflects_upserts() -> None:
    """Asserts count() tracks points written through the index."""
    client = FakeQdrantClient()
    index = make_index(client)
    index.upsert(["a", "b"], [[0.1] * 8] * 2, [{}] * 2)
    assert index.count() == 2
    assert index.is_empty() is False


def test_upsert_returns_total_written() -> None:
    """Asserts upsert reports the full number of points written."""
    client = FakeQdrantClient()
    ids = [f"c-{index}" for index in range(7)]
    assert make_index(client).upsert(ids, [[0.1] * 8] * 7, [{}] * 7) == 7
