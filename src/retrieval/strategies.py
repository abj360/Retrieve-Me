#!/usr/bin/env python3
"""
strategies.py --- pluggable retrieval strategy interface and legs

Contains:
    RetrievalStrategy: protocol all retrieval strategies implement
    QueryEmbedder: protocol for the embedder the dense leg injects
    VectorIndex: protocol for the vector store the dense leg searches
    Fuser: protocol for the fusion step the hybrid retriever injects
    Reranker: protocol for the rerank step the hybrid retriever injects
    SparseRetrievalStrategy: BM25 leg of the hybrid pipeline
    DenseRetrievalStrategy: dense vector leg of the hybrid pipeline
    HybridRetriever: orchestrates legs, fusion, and rerank into one call
    HybridRetriever.warmup(): loads both models before the first query
    STRATEGY_REGISTRY: strategy names to implementation classes (hybrid default)
"""

import logging
from typing import Protocol

import numpy as np

from src.ingest.bm25_index import BM25Index
from src.ingest.dense_index import DenseHit
from src.retrieval.fusion import RankedResult
from src.retrieval.tracing import LatencyTracer

logger = logging.getLogger(__name__)


class QueryEmbedder(Protocol):
    """Encodes query text into a dense vector."""

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes one query into a dense vector.

        Args:
            query: Raw query text.

        Returns:
            vector: Dense vector for the query.
        """

    def warmup(self) -> None:
        """Loads the embedding model before the first query arrives."""


class VectorIndex(Protocol):
    """Searches stored vectors for the nearest neighbours of a query."""

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
        score_threshold: float | None = None,
    ) -> list[DenseHit]:
        """Returns the nearest stored vectors to the query vector.

        Args:
            vector: Query vector.
            top_k: Maximum number of hits to return.
            filters: Optional exact-match payload filters.
            score_threshold: Minimum score for a hit to be returned.

        Returns:
            hits: Scored dense hits, best first.
        """


class Fuser(Protocol):
    """Merges the sparse and dense legs into one ranking."""

    def fuse(self, sparse: list[RankedResult], dense: list[RankedResult]) -> list[RankedResult]:
        """Merges two ranked lists into one fused ranking.

        Args:
            sparse: Ranked results from the BM25 leg.
            dense: Ranked results from the dense leg.

        Returns:
            fused: Deduplicated results in fused rank order.
        """


class Reranker(Protocol):
    """Re-scores fused candidates against the query."""

    def rerank(self, query: str, candidates: list[RankedResult]) -> list[RankedResult]:
        """Re-scores candidates and returns them best first.

        Args:
            query: Raw query text.
            candidates: Fused candidates to re-score.

        Returns:
            reranked: Re-scored candidates, best first.
        """

    def warmup(self) -> None:
        """Loads the reranker model before the first query arrives."""


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

    def __init__(self, index: VectorIndex, embedder: QueryEmbedder) -> None:
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
        query_vector = [float(value) for value in self.embedder.encode_query(query)]
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
        fuser: Fuser,
        reranker: Reranker,
        candidate_k: int = 50,
        tracer: LatencyTracer | None = None,
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
            fused = self.fuser.fuse(sparse_hits, dense_hits)  # legs normalized inside fuse
        with self.tracer.span("rerank"):
            return self.reranker.rerank(query, fused)[:top_k]

    def warmup(self) -> None:
        """Loads both models up front so the first query does not pay for them.

        Model loads dominate cold-start latency, so this runs at startup rather
        than on the query path. A failure here is logged and left alone: a slow
        first query is a better outcome than a service that refuses to start.
        """
        for stage, model in (("embedder", self.dense.embedder), ("reranker", self.reranker)):
            try:
                model.warmup()
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("%s warmup failed, first query will be cold: %s", stage, exc)
            else:
                logger.info("%s warmed", stage)

    def retrieve_with_candidate_k(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[RankedResult]:
        """Runs the hybrid pipeline with a per-query candidate_k override.

        Args:
            query: Raw query text.
            top_k: Final number of results to return.
            candidate_k: Candidate pool size for this query, or the default.
            filters: Optional exact-match metadata filters.

        Returns:
            results: Reranked results, best first, at most top_k.
        """
        original = self.candidate_k
        if candidate_k is not None:
            self.candidate_k = candidate_k
        try:
            return self.retrieve(query, top_k=top_k, filters=filters)
        finally:
            self.candidate_k = original  # restore even on failure


STRATEGY_REGISTRY: dict[str, type] = {
    "sparse": SparseRetrievalStrategy,
    "dense": DenseRetrievalStrategy,
    "hybrid": HybridRetriever,  # default for the config-driven pipeline
}
