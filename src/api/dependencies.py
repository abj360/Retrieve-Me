#!/usr/bin/env python3
"""
dependencies.py --- dependency-injection container for settings and clients

Contains:
    Settings: environment-driven service configuration
    get_settings(): returns the shared Settings instance
    get_redis(): builds a Redis connection from settings
    get_query_cache(): builds the shared query-response cache
"""

import redis
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.api.cache import QueryCache, RedisQueryCache


class Settings(BaseSettings):
    """Holds environment-driven configuration for the retrieval service.

    Attributes:
        qdrant_url: Base URL of the Qdrant instance.
        qdrant_collection: Collection that stores chunk vectors.
        redis_url: Connection URL for the Redis cache.
        cache_ttl_seconds: Time-to-live for cached query responses.
    """

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 300


def get_settings() -> Settings:
    """Returns the service settings, resolved from RETRIEVAL_* environment variables.

    Returns:
        settings: Populated Settings instance.
    """
    return Settings()


def get_redis(settings: Settings) -> redis.Redis:
    """Builds a Redis connection from settings.

    Args:
        settings: Service settings carrying the Redis URL.

    Returns:
        client: Redis client; connections open lazily on first use.
    """
    return redis.from_url(settings.redis_url, decode_responses=True)


def get_query_cache(settings: Settings | None = None) -> QueryCache:
    """Builds the query-response cache used by the retrieval endpoint.

    Args:
        settings: Service settings; loaded from the environment when omitted.

    Returns:
        cache: Redis-backed query cache with the configured TTL.
    """
    resolved = settings or get_settings()
    return RedisQueryCache(get_redis(resolved), ttl_seconds=resolved.cache_ttl_seconds)
