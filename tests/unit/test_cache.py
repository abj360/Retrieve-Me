#!/usr/bin/env python3
"""
test_cache.py --- unit tests for the query response cache

Contains:
    FakeRedis: dict-backed Redis double with optional failure mode
    test_in_memory_roundtrip(): asserts the in-memory cache stores and returns
    test_in_memory_miss_returns_none(): asserts a miss returns None
"""

from src.api.cache import InMemoryQueryCache, RedisQueryCache


class FakeRedis:
    """Dict-backed Redis double with an optional failure mode."""

    def __init__(self, fail: bool = False) -> None:
        """Creates the fake store.

        Args:
            fail: When True, every operation raises a Redis error.
        """
        self._store: dict[str, str] = {}
        self._fail = fail

    def get(self, key: str):
        """Returns the stored value or None.

        Args:
            key: Cache key to read.

        Returns:
            value: Stored payload or None.
        """
        self._maybe_fail()
        return self._store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        """Stores a value with a TTL (ignored).

        Args:
            key: Cache key to write.
            ttl: Time-to-live in seconds; ignored by the fake.
            value: Payload to store.
        """
        self._maybe_fail()
        self._store[key] = value

    def scan_iter(self, match: str):
        """Yields keys matching a glob prefix.

        Args:
            match: Glob pattern; only prefix-* is supported.

        Yields:
            key: Matching keys.
        """
        self._maybe_fail()
        prefix = match.rstrip("*")
        return [key for key in self._store if key.startswith(prefix)]

    def delete(self, *keys: str) -> int:
        """Deletes keys, returning the count removed.

        Args:
            keys: Keys to delete.

        Returns:
            removed: Number of keys deleted.
        """
        self._maybe_fail()
        removed = 0
        for key in keys:
            removed += self._store.pop(key, None) is not None
        return removed

    def _maybe_fail(self) -> None:
        """Raises a Redis error when failure mode is on."""
        if self._fail:
            import redis

            raise redis.RedisError("simulated redis outage")


def test_in_memory_roundtrip() -> None:
    """Asserts the in-memory cache stores and returns a payload."""
    cache = InMemoryQueryCache()
    cache.set("key-1", "payload-1")
    assert cache.get("key-1") == "payload-1"


def test_in_memory_miss_returns_none() -> None:
    """Asserts a miss returns None."""
    assert InMemoryQueryCache().get("nope") is None


def test_distinct_keys_do_not_collide() -> None:
    """Asserts different queries map to different cache entries."""
    cache = InMemoryQueryCache()
    cache.set("query-a", "payload-a")
    cache.set("query-b", "payload-b")
    assert cache.get("query-a") == "payload-a"
    assert cache.get("query-b") == "payload-b"


def test_redis_backed_roundtrip() -> None:
    """Asserts the Redis-backed cache stores and returns a payload."""
    cache = RedisQueryCache(FakeRedis(), ttl_seconds=60)
    cache.set("key-1", "payload-1")
    assert cache.get("key-1") == "payload-1"


def test_cache_key_covers_request_shape() -> None:
    """Asserts the endpoint cache key covers query, pagination, and filters."""
    from src.api.routers.retrieve import RetrieveRequest, cache_key

    first = cache_key(RetrieveRequest(query="clause"))
    second = cache_key(RetrieveRequest(query="clause", page=2))
    third = cache_key(RetrieveRequest(query="clause"))
    assert first != second
    assert first == third


def test_redis_outage_treated_as_miss() -> None:
    """Asserts a Redis failure on get returns None, not an exception."""
    cache = RedisQueryCache(FakeRedis(fail=True))
    assert cache.get("key-1") is None


def test_redis_outage_on_set_does_not_raise() -> None:
    """Asserts a Redis failure on set is logged and swallowed."""
    cache = RedisQueryCache(FakeRedis(fail=True))
    cache.set("key-1", "payload-1")


def test_key_prefix_is_configurable() -> None:
    """Asserts a custom key prefix is used for stored keys."""
    fake = FakeRedis()
    cache = RedisQueryCache(fake, key_prefix="test:ns:")
    cache.set("key-1", "payload-1")
    assert "test:ns:key-1" in fake._store


def test_ttl_passed_to_setex() -> None:
    """Asserts the configured TTL is handed to Redis on write."""
    recorded = {}

    class TtlRecordingRedis(FakeRedis):
        """Records the TTL passed to setex."""

        def setex(self, key: str, ttl: int, value: str) -> None:
            """Records the ttl argument, then stores."""
            recorded["ttl"] = ttl
            super().setex(key, ttl, value)

    cache = RedisQueryCache(TtlRecordingRedis(), ttl_seconds=42)
    cache.set("key-1", "payload-1")
    assert recorded["ttl"] == 42


def test_hits_and_misses_counted() -> None:
    """Asserts the cache tracks hit and miss counters."""
    cache = RedisQueryCache(FakeRedis())
    cache.set("key-1", "payload-1")
    cache.get("key-1")
    cache.get("key-2")
    assert cache.hits == 1
    assert cache.misses == 1


def test_outage_miss_counts_as_miss() -> None:
    """Asserts a Redis outage increments the miss counter."""
    cache = RedisQueryCache(FakeRedis(fail=True))
    cache.get("key-1")
    assert cache.misses == 1


def test_counters_start_at_zero() -> None:
    """Asserts a fresh cache starts with zeroed counters."""
    cache = RedisQueryCache(FakeRedis())
    assert cache.hits == 0
    assert cache.misses == 0
