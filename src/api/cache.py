#!/usr/bin/env python3
"""
cache.py --- Redis-backed cache for repeated retrieval queries

Contains:
    QueryCache: interface for query-response caches
    RedisQueryCache: stores serialized responses in Redis with a TTL
    InMemoryQueryCache: dict-backed cache for tests and local dev
"""

import logging
from typing import Protocol

import redis

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300
KEY_PREFIX = "Retrieve-Me:query:"


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
    """

    def __init__(self, client: redis.Redis, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """Stores the client and TTL; connections open lazily.

        Args:
            client: Redis connection used for reads and writes.
            ttl_seconds: Time-to-live applied to every cached entry.
        """
        self._client = client
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> str | None:
        """Returns the cached payload for key, or None on a miss.

        Args:
            key: Canonical cache key for one retrieval request.

        Returns:
            payload: Cached serialized response, or None.
        """
        value = self._client.get(KEY_PREFIX + key)
        return value if isinstance(value, str) else None

    def set(self, key: str, payload: str) -> None:
        """Stores a serialized response under key with the configured TTL.

        Args:
            key: Canonical cache key for one retrieval request.
            payload: Serialized response body to cache.
        """
        self._client.setex(KEY_PREFIX + key, self.ttl_seconds, payload)


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
