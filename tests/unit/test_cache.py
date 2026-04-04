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
