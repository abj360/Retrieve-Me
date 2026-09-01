#!/usr/bin/env python3
"""
dependencies.py --- dependency-injection container for settings and clients

Contains:
    Settings: environment-driven service configuration
    get_settings(): returns the shared Settings instance
    get_redis(): builds a Redis connection from settings
    build_query_cache(): builds a query-response cache from explicit settings
    get_query_cache(): returns the shared query-response cache
    get_qdrant_pool(): builds the shared Qdrant client pool
    build_pipeline(): assembles the hybrid pipeline from pipeline.yaml
    get_pipeline(): returns the shared hybrid retrieval pipeline
"""

from functools import lru_cache

import redis
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.api.cache import QueryCache, RedisQueryCache
from src.config.pipeline import load_pipeline_config
from src.ingest.bm25_index import BM25Index
from src.ingest.dense_index import DenseIndex, QdrantClientPool, QdrantConfig
from src.retrieval.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from src.retrieval.fusion import FusionConfig, ResultFuser
from src.retrieval.rerank import CrossEncoderReranker, RerankerConfig
from src.retrieval.strategies import (
    DenseRetrievalStrategy,
    HybridRetriever,
    SparseRetrievalStrategy,
)

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_K = 50
DEFAULT_CACHE_TTL_SECONDS = 300
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_REQUEST_TIMEOUT = 30.0


class Settings(BaseSettings):
    """Holds environment-driven configuration for the retrieval service.

    Attributes:
        qdrant_url: Base URL of the Qdrant instance.
        qdrant_collection: Collection that stores chunk vectors.
        redis_url: Connection URL for the Redis cache.
        cache_ttl_seconds: Time-to-live for cached query responses.
        default_top_k: Default number of chunks returned per query.
        candidate_k: Number of candidates fused before reranking.
        log_level: Root log level for the service.
        embedding_model: Sentence-transformers model for dense embeddings.
        reranker_model: Cross-encoder model used for reranking.
        reranker_top_k: Candidates kept after cross-encoder reranking (tuned).
        fusion_rrf_k: Reciprocal-rank-fusion smoothing constant.
        pipeline_config_path: Path to the YAML pipeline definition.
        app_title: Human-readable service title for the OpenAPI docs.
        app_version: Service version reported by the API and /healthz.
    """

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    qdrant_url: str = "http://localhost:6333"  # override with RETRIEVAL_QDRANT_URL
    qdrant_collection: str = "chunks"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    default_top_k: int = DEFAULT_TOP_K
    candidate_k: int = DEFAULT_CANDIDATE_K
    log_level: str = DEFAULT_LOG_LEVEL
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 20
    fusion_rrf_k: int = 60
    pipeline_config_path: str = "src/config/pipeline.yaml"
    app_title: str = "Retrieve-Me"
    app_version: str = "1.1.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the service settings, resolved from RETRIEVAL_* environment variables.

    Returns:
        settings: Cached Settings instance shared by all providers.
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


def build_query_cache(settings: Settings) -> QueryCache:
    """Builds the query-response cache from explicit settings.

    Args:
        settings: Service settings carrying the Redis URL and cache TTL.

    Returns:
        cache: Redis-backed query cache with the configured TTL.
    """
    return RedisQueryCache(get_redis(settings), ttl_seconds=settings.cache_ttl_seconds)


def get_query_cache() -> QueryCache:
    """Returns the query-response cache injected into the retrieval endpoint.

    Takes no arguments: FastAPI reads a provider's signature to build the
    request contract, so a settings parameter here becomes a request body field.

    Returns:
        cache: Cache built from the shared settings.
    """
    return build_query_cache(get_settings())


def get_qdrant_pool(settings: Settings) -> QdrantClientPool:
    """Builds the shared Qdrant client pool.

    Args:
        settings: Service settings carrying the Qdrant URL and collection.

    Returns:
        pool: Bounded pool of Qdrant clients.
    """
    config = QdrantConfig(url=settings.qdrant_url, collection=settings.qdrant_collection)
    return QdrantClientPool(config)


def build_pipeline(settings: Settings) -> HybridRetriever:
    """Assembles the hybrid retrieval pipeline from settings.

    Args:
        settings: Service settings with model names and pool targets.

    Returns:
        pipeline: Hybrid retriever combining sparse, dense, fusion, rerank.
    """
    config = load_pipeline_config(settings.pipeline_config_path)
    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name=config.embedder.model_name,
            batch_size=config.embedder.batch_size,
        )
    )
    sparse = SparseRetrievalStrategy(BM25Index())
    dense_index = DenseIndex(
        QdrantConfig(url=settings.qdrant_url, collection=settings.qdrant_collection),
        get_qdrant_pool(settings),
    )
    dense = DenseRetrievalStrategy(dense_index, embedder)
    fuser = ResultFuser(
        FusionConfig(
            rrf_k=config.fusion.rrf_k,
            sparse_weight=config.fusion.sparse_weight,
            dense_weight=config.fusion.dense_weight,
            normalize_scores=config.fusion.normalize_scores,
        )
    )
    reranker = CrossEncoderReranker(
        RerankerConfig(
            model_name=config.reranker.model_name,
            top_k=config.reranker.top_k,
            batch_size=config.reranker.batch_size,
            min_score=config.reranker.min_score,
        )
    )
    return HybridRetriever(sparse, dense, fuser, reranker, candidate_k=config.strategy.candidate_k)


@lru_cache(maxsize=1)
def get_pipeline() -> HybridRetriever:
    """Returns the shared hybrid retrieval pipeline, built once and cached.

    Returns:
        pipeline: Pipeline assembled from the cached settings.
    """
    return build_pipeline(get_settings())
