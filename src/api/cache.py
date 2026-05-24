#!/usr/bin/env python3
"""
cache.py --- Redis-backed cache for repeated retrieval queries

Contains:
    QueryCache: interface for query-response caches (get/set by canonical key)
    RedisQueryCache: stores serialized responses in Redis with a TTL
    RedisQueryCache.clear(): deletes every entry under the key prefix
    InMemoryQueryCache: dict-backed cache for tests and local dev
"""

import logging
from typing import Protocol

import redis

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300  # five minutes covers most repeated benchmarks
KEY_PREFIX = "retrieve-me:q:"


class QueryCache(Protocol):
    """Stores and retrieves serialized query responses by key."""

    def get(self, key: str) -> str | None:
        """Returns the cached payload for key, or None on a miss.

        Args:
            key: Canonical cache key for one retrieval request.

        Returns:
            payload: Cached serialized response, or None.
        """

    def set(self, key: str, payload: str) -> None:
        """Stores a serialized response under key.

        Args:
            key: Canonical cache key for one retrieval request.
            payload: Serialized response body to cache.
        """


class RedisQueryCache:
    """Stores serialized responses in Redis with a TTL.

    Attributes:
        ttl_seconds: Time-to-live applied to every cached entry.
        hits: Number of cache hits since startup.
        misses: Number of cache misses since startup.
    """

    def __init__(
        self,
        client: redis.Redis,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = KEY_PREFIX,
    ) -> None:
        """Stores the client, TTL, and key prefix; connections open lazily.

        Args:
            client: Redis connection used for reads and writes.
            ttl_seconds: Time-to-live applied to every cached entry.
            key_prefix: Namespace prefix for every stored key.
        """
        self._client = client
        self.ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        """Returns the cached payload for key, or None on a miss.

        Args:
            key: Canonical cache key for one retrieval request.

        Returns:
            payload: Cached serialized response, or None.
        """
        try:
            value = self._client.get(self._key_prefix + key)
        except redis.RedisError as exc:
            logger.warning("cache get failed, serving request as a miss: %s", exc)
            self.misses += 1
            return None
        if isinstance(value, str):
            self.hits += 1
            return value
        self.misses += 1
        return None

    def set(self, key: str, payload: str) -> None:
        """Stores a serialized response under key with the configured TTL.

        Args:
            key: Canonical cache key for one retrieval request.
            payload: Serialized response body to cache.
        """
        try:
            self._client.setex(self._key_prefix + key, self.ttl_seconds, payload)
        except redis.RedisError as exc:
            logger.warning("cache set failed, response not cached: %s", exc)

    def clear(self) -> int:
        """Deletes every cached entry under the key prefix.

        Returns:
            removed: Number of keys deleted.
        """
        keys = list(self._client.scan_iter(match=f"{self._key_prefix}*"))
        if not keys:
            return 0
        return int(self._client.delete(*keys))


class InMemoryQueryCache:
    """Stores serialized responses in a dict for tests and local dev."""

    def __init__(self) -> None:
        """Creates an empty in-memory cache."""
        self._entries: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Returns the cached payload for key, or None on a miss.

        Args:
            key: Canonical cache key for one retrieval request.

        Returns:
            payload: Cached serialized response, or None.
        """
        return self._entries.get(KEY_PREFIX + key)

    def set(self, key: str, payload: str) -> None:
        """Stores a serialized response under key.

        Args:
            key: Canonical cache key for one retrieval request.
            payload: Serialized response body to cache.
        """
        self._entries[KEY_PREFIX + key] = payload
