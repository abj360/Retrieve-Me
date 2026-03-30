#!/usr/bin/env python3
"""
dependencies.py --- dependency-injection container for settings and clients

Contains:
    Settings: environment-driven service configuration
    get_settings(): returns the shared Settings instance
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Holds environment-driven configuration for the retrieval service.

    Attributes:
        qdrant_url: Base URL of the Qdrant instance.
        qdrant_collection: Collection that stores chunk vectors.
        redis_url: Connection URL for the Redis cache.
    """

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    redis_url: str = "redis://localhost:6379"


def get_settings() -> Settings:
    """Returns the service settings, built from the environment.

    Returns:
        settings: Populated Settings instance.
    """
    return Settings()
