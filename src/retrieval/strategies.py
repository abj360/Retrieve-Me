#!/usr/bin/env python3
"""
strategies.py --- pluggable retrieval strategy interface and legs

Contains:
    RetrievalStrategy: interface all retrieval strategies implement
    SparseRetrievalStrategy: BM25 leg of the hybrid pipeline
    DenseRetrievalStrategy: dense vector leg of the hybrid pipeline
    HybridRetriever: orchestrates legs, fusion, and rerank into one call
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


class HybridRetriever:
    """Orchestrates sparse, dense, fusion, and rerank into one retrieve call.

    Attributes:
        candidate_k: Candidates fused per leg before reranking.
    """

    def __init__(
        self,
        sparse: SparseRetrievalStrategy,
        dense: DenseRetrievalStrategy,
        fuser,
        reranker,
        candidate_k: int = 50,
        tracer=None,
    ) -> None:
        """Stores the pipeline stages.

        Args:
            sparse: Sparse retrieval leg.
            dense: Dense retrieval leg.
            fuser: RRF fuser for the two legs.
            reranker: Cross-encoder reranker for fused candidates.
            candidate_k: Candidates fused per leg before reranking.
            tracer: Optional latency tracer for per-stage spans.
        """
        self.sparse = sparse
        self.dense = dense
        self.fuser = fuser
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.tracer = tracer

    def retrieve(
        self, query: str, top_k: int = 10, filters: dict[str, str] | None = None
    ) -> list[RankedResult]:
        """Runs the full hybrid pipeline for one query.

        Args:
            query: Raw query text.
            top_k: Final number of results to return.
            filters: Optional exact-match metadata filters.

        Returns:
            results: Reranked results, best first, at most top_k.
        """
        if self.tracer is None:
            sparse_hits = self.sparse.retrieve(query, self.candidate_k)
            dense_hits = self.dense.retrieve(query, self.candidate_k, filters)
            fused = self.fuser.fuse(sparse_hits, dense_hits)
            return self.reranker.rerank(query, fused)[:top_k]
        with self.tracer.span("sparse"):
            sparse_hits = self.sparse.retrieve(query, self.candidate_k)
        with self.tracer.span("dense"):
            dense_hits = self.dense.retrieve(query, self.candidate_k, filters)
        with self.tracer.span("fuse"):
            fused = self.fuser.fuse(sparse_hits, dense_hits)
        with self.tracer.span("rerank"):
            return self.reranker.rerank(query, fused)[:top_k]
