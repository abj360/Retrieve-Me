#!/usr/bin/env python3
"""
dense_index.py --- Qdrant-backed dense vector index with a bounded client pool

Contains:
    QdrantConfig: connection and collection settings for the dense index
    PoolExhaustedError: raised when no client is free within the acquire timeout
    retry_with_backoff(): retries a transient Qdrant call with exponential backoff
    QdrantClientPool: lends a bounded set of Qdrant clients
    DenseHit: one scored hit from the dense index
    DenseIndex: builds and searches the Qdrant collection
"""

import logging
import queue
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output size
DEFAULT_BATCH_SIZE = 100  # qdrant handles this comfortably per request
DEFAULT_POOL_SIZE = 8  # one client per expected concurrent worker
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 5.0
MAX_RETRIES = 3  # transient blips clear within a second of backoff
RETRY_BASE_DELAY_SECONDS = 0.2  # doubles each attempt

T = TypeVar("T")


class PoolExhaustedError(RuntimeError):
    """Raised when the pool cannot lend a client before the acquire timeout."""


def retry_with_backoff(operation: Callable[[], T], retries: int = MAX_RETRIES) -> T:
    """Retries a transient Qdrant operation with exponential backoff.

    Args:
        operation: Zero-argument callable to attempt.
        retries: Maximum number of attempts before giving up.

    Returns:
        result: Whatever the operation returns on success.
    """
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except (UnexpectedResponse, TimeoutError) as exc:
            if attempt == retries:
                raise
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning("qdrant call failed (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError("retry loop exited unexpectedly")

@dataclass(frozen=True)
class QdrantConfig:
    """Carries connection and collection settings for the dense index (frozen).

    Attributes:
        url: Base URL of the Qdrant instance.
        collection: Name of the collection that stores chunk vectors.
        vector_size: Dimensionality of the stored vectors.
        timeout: Per-request timeout in seconds.
        distance: Distance metric used by the collection.
        distance: Distance metric used by the collection.
    """

    url: str = "http://localhost:6333"  # RETRIEVAL_QDRANT_URL overrides
    collection: str = "chunks"
    vector_size: int = DEFAULT_VECTOR_SIZE
    timeout: float = 10.0
    distance: Distance = Distance.COSINE  # cosine for normalized embeddings


@dataclass(frozen=True)
class DenseHit:
    """Carries one scored hit from the dense index (chunk id, cosine score, payload).

    Attributes:
        chunk_id: Stable identifier of the matched chunk.
        score: Cosine similarity score assigned by Qdrant.
        payload: Metadata stored alongside the vector.
    """

    chunk_id: str
    score: float
    payload: dict


class QdrantClientPool:
    """Lends a bounded set of Qdrant clients to concurrent callers.

    Attributes:
        config: Connection settings shared by pooled clients.
    """

    def __init__(
        self,
        config: QdrantConfig,
        max_size: int = DEFAULT_POOL_SIZE,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
    ) -> None:
        """Pre-creates max_size clients handed out under a blocking bound.

        Args:
            config: Connection settings shared by pooled clients.
            max_size: Maximum number of clients the pool hands out.
            acquire_timeout: Seconds a caller waits before PoolExhaustedError.
        """
        self.config = config
        self._acquire_timeout = acquire_timeout
        self._idle: queue.Queue[QdrantClient] = queue.Queue(maxsize=max_size)
        for _ in range(max_size):
            self._idle.put(QdrantClient(url=config.url, timeout=config.timeout))

    @contextmanager
    def acquire(self) -> Iterator[QdrantClient]:
        """Yields a pooled client, blocking up to the acquire timeout.

        Returns:
            client: Pooled Qdrant client the caller must yield back.
        """
        try:
            client = self._idle.get(timeout=self._acquire_timeout)
        except queue.Empty as exc:
            raise PoolExhaustedError(
                f"no Qdrant client free within {self._acquire_timeout}s"
            ) from exc
        try:
            yield client
        finally:
            self._idle.put(client)

    def close(self) -> None:
        """Closes every idle client held by the pool."""
        while not self._idle.empty():
            self._idle.get_nowait().close()


class DenseIndex:
    """Builds and searches the Qdrant collection for chunk vectors.

    Attributes:
        config: Collection and connection settings.
        pool: Client pool used for all Qdrant calls.
    """

    def __init__(self, config: QdrantConfig, pool: QdrantClientPool) -> None:
        """Stores config and pool without touching the server.

        Args:
            config: Collection and connection settings.
            pool: Client pool used for all Qdrant calls.
        """
        self.config = config
        self.pool = pool

    def ensure_collection(self, recreate: bool = False) -> None:
        """Creates the collection when it does not exist yet.

        Args:
            recreate: Drops and recreates the collection when True.
        """
        with self.pool.acquire() as client:
            exists = client.collection_exists(self.config.collection)
            if exists and recreate:
                client.delete_collection(self.config.collection)
                exists = False
            if exists:
                return
            client.create_collection(
                collection_name=self.config.collection,
                vectors_config=VectorParams(
                    size=self.config.vector_size, distance=self.config.distance
                ),
            )

    def count(self) -> int:
        """Counts the points currently stored in the collection.

        Returns:
            total: Number of points in the collection, 0 when missing.
        """
        with self.pool.acquire() as client:
            if not client.collection_exists(self.config.collection):
                logger.debug("count on missing collection, returning 0")
                return 0
            return int(client.count(self.config.collection).count)

    def is_empty(self) -> bool:
        """Returns whether the collection has no points.

        Returns:
            empty: True when the collection is missing or holds no points.
        """
        return self.count() == 0

    def upsert(
        self,
        chunk_ids: list[str],
        vectors: list,
        payloads: list[dict],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> int:
        """Upserts vectors with their payloads in batches.

        Args:
            chunk_ids: Stable chunk identifiers, one per vector.
            vectors: Dense vectors to store.
            payloads: Metadata stored alongside each vector.
            batch_size: Maximum number of points per upsert request.

        Returns:
            upserted: Number of points written.
        """
        points = [
            PointStruct(id=index, vector=vector, payload={"chunk_id": chunk_id, **payload})
            for index, (chunk_id, vector, payload) in enumerate(
                zip(chunk_ids, vectors, payloads, strict=True)
            )
        ]
        if not points:
            logger.debug("upsert called with no points, skipping")
            return 0
        with self.pool.acquire() as client:
            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]
                retry_with_backoff(
                    lambda: client.upsert(
                        collection_name=self.config.collection, points=batch
                    )
                )
                logger.debug(
                    "upserted batch %d-%d of %d points",
                    start,
                    start + len(batch),
                    len(points),
                )
        return len(points)

    def search(self, vector: list[float], top_k: int) -> list[DenseHit]:
        """Searches the collection for the nearest vectors.

        Args:
            vector: Query vector.
            top_k: Maximum number of hits to return.

        Returns:
            hits: Scored dense hits, best first.
        """
        if self.is_empty():
            logger.debug("dense search skipped: collection empty")
            return []
        with self.pool.acquire() as client:
            try:
                response = retry_with_backoff(
                    lambda: client.query_points(
                        collection_name=self.config.collection,
                        query=vector,
                        limit=top_k,
                    )
                )
            except UnexpectedResponse as exc:
                logger.warning("dense search hit a missing collection: %s", exc)
                return []
        return [
            DenseHit(
                chunk_id=point.payload["chunk_id"],
                score=point.score,
                payload=point.payload,
            )
            for point in response.points
        ]
