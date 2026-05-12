#!/usr/bin/env python3
"""
conftest.py --- shared fixtures for the retrieval-core test suites

Contains:
    TestChunk: minimal chunk stand-in used before the chunking module landed
    sample_documents(): small legal/tech corpus used across suites
    sample_chunks(): chunks built from the sample documents
    whole_doc_chunker(): test chunker that keeps each document whole
    stub_embedder(): deterministic hashing embedder for offline tests
    fake_dense_index(): in-memory dense index double
    bm25_index(): sparse index built over the sample chunks
    token_chunker(): real token-aware chunker with a small test budget
    stub_reranker(): deterministic reranker double scoring by text overlap
    real_embedder(): the shared DeterministicEmbedder used by smoke tests
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pytest

from src.ingest.bm25_index import BM25Index
from src.ingest.chunking import ChunkConfig, SemanticClauseChunker
from src.ingest.dense_index import DenseHit
from src.retrieval.embeddings import DeterministicEmbedder


@dataclass(frozen=True)
class TestChunk:
    """Carries the chunk fields the pipeline needs in tests.

    Attributes:
        chunk_id: Stable identifier of the chunk.
        doc_id: Identifier of the document the chunk came from.
        text: Raw chunk text.
        token_count: Number of tokens in the chunk.
        index: Position of the chunk within the document.
        metadata: Extra attributes carried through to the indexes.
    """

    chunk_id: str
    doc_id: str
    text: str
    token_count: int = 0
    index: int = 0
    metadata: dict = field(default_factory=dict)


class WholeDocChunker:
    """Keeps each document as a single chunk for pipeline tests."""

    def split(self, text: str, doc_id: str) -> list[TestChunk]:
        """Returns the whole document as one chunk.

        Args:
            text: Raw document text.
            doc_id: Identifier of the document being split.

        Returns:
            chunks: Single-chunk list covering the document.
        """
        return [
            TestChunk(
                chunk_id=f"{doc_id}-chunk-0",
                doc_id=doc_id,
                text=text,
                token_count=len(text.split()),
            )
        ]


class StubEmbedder:
    """Produces deterministic vectors from text hashes for offline tests.

    Attributes:
        dimension: Dimensionality of the produced vectors.
    """

    def __init__(self, dimension: int = 16) -> None:
        """Stores the vector dimension.

        Args:
            dimension: Dimensionality of the produced vectors.
        """
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encodes texts into deterministic pseudo-embeddings.

        Args:
            texts: Raw texts to encode.

        Returns:
            vectors: One normalized vector per input text.
        """
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode()).digest()
            vectors[row] = np.frombuffer(digest, dtype=np.uint8)[: self.dimension] / 255.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


class FakeDenseIndex:
    """Keeps vectors in memory and answers search with real cosine scores."""

    def __init__(self) -> None:
        """Creates an empty in-memory store."""
        self._store: dict[str, tuple[np.ndarray, dict]] = {}

    def ensure_collection(self, recreate: bool = False) -> None:
        """Clears the store when recreate is requested.

        Args:
            recreate: Whether to drop all stored vectors.
        """
        if recreate:
            self._store.clear()

    def upsert(
        self,
        chunk_ids: list[str],
        vectors: list,
        payloads: list[dict],
        batch_size: int = 100,
    ) -> int:
        """Stores vectors and payloads in memory.

        Args:
            chunk_ids: Stable chunk identifiers, one per vector.
            vectors: Dense vectors to store.
            payloads: Metadata stored alongside each vector.
            batch_size: Ignored; present for interface compatibility.

        Returns:
            upserted: Number of points written.
        """
        for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads, strict=True):
            self._store[chunk_id] = (np.asarray(vector, dtype=np.float32), payload)
        return len(chunk_ids)

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
        score_threshold: float | None = None,
    ) -> list[DenseHit]:
        """Returns the top_k stored vectors by cosine similarity.

        Args:
            vector: Query vector.
            top_k: Maximum number of hits to return.
            filters: Optional exact-match payload filters.
            score_threshold: Minimum cosine score for a hit to be returned.

        Returns:
            hits: Scored dense hits, best first.
        """
        query = np.asarray(vector, dtype=np.float32)
        scored = []
        for chunk_id, (stored, payload) in self._store.items():
            if filters and any(payload.get(field) != value for field, value in filters.items()):
                continue
            denominator = float(np.linalg.norm(query) * np.linalg.norm(stored))
            score = float(np.dot(query, stored) / denominator) if denominator else 0.0
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(DenseHit(chunk_id=chunk_id, score=score, payload=payload))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """Counts the vectors held in memory.

        Returns:
            total: Number of stored vectors.
        """
        return len(self._store)

    def is_empty(self) -> bool:
        """Returns whether the store holds no vectors.

        Returns:
            empty: True when nothing is stored.
        """
        return not self._store


@pytest.fixture
def sample_documents() -> list[tuple[str, str]]:
    """Returns a small legal/tech corpus as (doc_id, text) pairs."""
    return [
        (
            "license-agreement",
            "Section 3.1 The licensee shall indemnify the vendor against claims. "
            "This obligation survives termination of the agreement. "
            "Section 3.2 The vendor provides the software as is, without warranty.",
        ),
        (
            "rfc-7807",
            "RFC 7807 defines problem details for HTTP APIs. "
            "The problem detail response carries a type, title, status, and detail member.",
        ),
        (
            "release-notes",
            "Release 2.4 adds batch upserts to the vector store client. "
            "Connection pooling is now bounded to prevent exhaustion under load.",
        ),
        (
            "memo-q3",
            "Q3 planning memo: the retrieval team will benchmark hybrid search "
            "against the dense-only baseline on the legal corpus.",
        ),
    ]


@pytest.fixture
def whole_doc_chunker() -> WholeDocChunker:
    """Returns the test chunker that keeps each document whole."""
    return WholeDocChunker()


@pytest.fixture
def sample_chunks(sample_documents, whole_doc_chunker) -> list[TestChunk]:
    """Builds chunks from the sample documents."""
    return [
        chunk
        for doc_id, text in sample_documents
        for chunk in whole_doc_chunker.split(text, doc_id)
    ]


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    """Returns the deterministic hashing embedder."""
    return StubEmbedder()


@pytest.fixture
def fake_dense_index() -> FakeDenseIndex:
    """Returns an empty in-memory dense index double."""
    return FakeDenseIndex()


@pytest.fixture
def token_chunker() -> SemanticClauseChunker:
    """Returns the real chunker with a small token budget for tests."""
    return SemanticClauseChunker(ChunkConfig(max_tokens=48, overlap_tokens=8, min_chunk_tokens=5))


class StubReranker:
    """Scores candidates by token overlap with the query, deterministically."""

    def rerank(self, query: str, candidates: list) -> list:
        """Re-scores candidates by query token overlap.

        Args:
            query: Raw query text.
            candidates: Ranked candidates from the fusion step.

        Returns:
            reranked: Candidates sorted by overlap score, best first.
        """
        query_terms = set(query.lower().split())
        scored = []
        for candidate in candidates:
            overlap = len(query_terms & set(candidate.text.lower().split()))
            scored.append((overlap, candidate))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [candidate for _, candidate in scored]


@pytest.fixture
def stub_reranker() -> StubReranker:
    """Returns the deterministic reranker double."""
    return StubReranker()


@pytest.fixture
def real_embedder() -> DeterministicEmbedder:
    """Returns the shared deterministic embedder for smoke tests."""
    return DeterministicEmbedder(dimension=16)


@pytest.fixture
def bm25_index(sample_chunks) -> BM25Index:
    """Builds a sparse index over the sample chunks."""
    index = BM25Index()
    index.build(sample_chunks)
    return index
