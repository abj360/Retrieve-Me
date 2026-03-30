#!/usr/bin/env python3
"""
dense_index.py --- Qdrant-backed dense vector index with client pooling

Contains:
    QdrantConfig: connection and collection settings for the dense index
    QdrantClientPool: hands out Qdrant clients to callers
    DenseHit: one scored hit from the dense index
    DenseIndex: builds and searches the Qdrant collection
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_SIZE = 384


@dataclass(frozen=True)
class QdrantConfig:
    """Carries connection and collection settings for the dense index.

    Attributes:
        url: Base URL of the Qdrant instance.
        collection: Name of the collection that stores chunk vectors.
        vector_size: Dimensionality of the stored vectors.
        timeout: Per-request timeout in seconds.
    """

    url: str = "http://localhost:6333"
    collection: str = "chunks"
    vector_size: int = DEFAULT_VECTOR_SIZE
    timeout: float = 10.0


@dataclass(frozen=True)
class DenseHit:
    """Carries one scored hit from the dense index.

    Attributes:
        chunk_id: Stable identifier of the matched chunk.
        score: Cosine similarity score assigned by Qdrant.
        payload: Metadata stored alongside the vector.
    """

    chunk_id: str
    score: float
    payload: dict


class QdrantClientPool:
    """Lends Qdrant clients to callers, opening connections on demand.

    Attributes:
        config: Connection settings shared by lent clients.
    """

    def __init__(self, config: QdrantConfig) -> None:
        """Stores config; clients are created lazily as callers arrive.

        Args:
            config: Connection settings shared by lent clients.
        """
        self.config = config
        self._idle: list[QdrantClient] = []

    @contextmanager
    def acquire(self) -> Iterator[QdrantClient]:
        """Yields a client, opening a fresh connection when none are idle.

        Returns:
            client: Qdrant client the caller must yield back.
        """
        client = self._idle.pop() if self._idle else QdrantClient(url=self.config.url)
        try:
            yield client
        finally:
            self._idle.append(client)


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

    def ensure_collection(self) -> None:
        """Creates the collection when it does not exist yet."""
        with self.pool.acquire() as client:
            if client.collection_exists(self.config.collection):
                return
            client.create_collection(
                collection_name=self.config.collection,
                vectors_config=VectorParams(
                    size=self.config.vector_size, distance=Distance.COSINE
                ),
            )

    def upsert(self, chunk_ids: list[str], vectors: list, payloads: list[dict]) -> int:
        """Upserts vectors with their payloads into the collection.

        Args:
            chunk_ids: Stable chunk identifiers, one per vector.
            vectors: Dense vectors to store.
            payloads: Metadata stored alongside each vector.

        Returns:
            upserted: Number of points written.
        """
        points = [
            PointStruct(id=index, vector=vector, payload={"chunk_id": chunk_id, **payload})
            for index, (chunk_id, vector, payload) in enumerate(
                zip(chunk_ids, vectors, payloads, strict=True)
            )
        ]
        with self.pool.acquire() as client:
            client.upsert(collection_name=self.config.collection, points=points)
        return len(points)

    def search(self, vector: list[float], top_k: int) -> list[DenseHit]:
        """Searches the collection for the nearest vectors.

        Args:
            vector: Query vector.
            top_k: Maximum number of hits to return.

        Returns:
            hits: Scored dense hits, best first.
        """
        with self.pool.acquire() as client:
            response = client.query_points(
                collection_name=self.config.collection,
                query=vector,
                limit=top_k,
            )
        return [
            DenseHit(
                chunk_id=point.payload["chunk_id"],
                score=point.score,
                payload=point.payload,
            )
            for point in response.points
        ]
