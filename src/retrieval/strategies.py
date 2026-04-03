#!/usr/bin/env python3
"""
strategies.py --- pluggable retrieval strategy interface and legs

Contains:
    RetrievalStrategy: interface all retrieval strategies implement
    SparseRetrievalStrategy: BM25 leg of the hybrid pipeline
    DenseRetrievalStrategy: dense vector leg of the hybrid pipeline
"""

import logging
from typing import Protocol

from src.ingest.bm25_index import BM25Index
from src.retrieval.fusion import RankedResult

logger = logging.getLogger(__name__)


class RetrievalStrategy(Protocol):
    """Retrieves ranked results for a query."""

    def retrieve(
        self, query: str, top_k: int, filters: dict[str, str] | None = None
    ) -> list[RankedResult]:
        """Retrieves ranked results for a query.

        Args:
            query: Raw query text.
            top_k: Maximum number of results to return.
            filters: Optional exact-match metadata filters.

        Returns:
            results: Ranked results, best first.
        """


class SparseRetrievalStrategy:
    """Retrieves ranked results from the BM25 sparse index.

    Attributes:
        index: Sparse index searched per query.
    """

    def __init__(self, index: BM25Index) -> None:
        """Stores the sparse index.

        Args:
            index: Sparse index searched per query.
        """
        self.index = index

    def retrieve(
        self, query: str, top_k: int, filters: dict[str, str] | None = None
    ) -> list[RankedResult]:
        """Retrieves ranked results from the sparse index.

        Args:
            query: Raw query text.
            top_k: Maximum number of results to return.
            filters: Ignored by the sparse leg.

        Returns:
            results: Sparse hits mapped to ranked results.
        """
        return [
            RankedResult(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                text=hit.text,
                score=hit.score,
                source="sparse",
            )
            for hit in self.index.search(query, top_k)
        ]


class DenseRetrievalStrategy:
    """Retrieves ranked results from the dense vector index.

    Attributes:
        index: Dense index searched per query.
        embedder: Embedder used to vectorize queries.
    """

    def __init__(self, index, embedder) -> None:
        """Stores the dense index and embedder.

        Args:
            index: Dense index searched per query.
            embedder: Embedder used to vectorize queries.
        """
        self.index = index
        self.embedder = embedder

    def retrieve(
        self, query: str, top_k: int, filters: dict[str, str] | None = None
    ) -> list[RankedResult]:
        """Retrieves ranked results from the dense index.

        Args:
            query: Raw query text.
            top_k: Maximum number of results to return.
            filters: Optional exact-match payload filters.

        Returns:
            results: Dense hits mapped to ranked results.
        """
        query_vector = self.embedder.encode_query(query)
        return [
            RankedResult(
                chunk_id=hit.chunk_id,
                doc_id=hit.payload["doc_id"],
                text=hit.payload["text"],
                score=hit.score,
                source="dense",
            )
            for hit in self.index.search(query_vector, top_k, filters=filters)
        ]
