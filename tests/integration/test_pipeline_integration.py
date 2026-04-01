#!/usr/bin/env python3
"""
test_pipeline_integration.py --- integration tests for the retrieval pipeline

Contains:
    indexed_stores(): ingests the sample corpus into fake stores
    test_exact_clause_doc_wins_fused_ranking(): asserts hybrid beats dense-only
    test_sparse_leg_finds_exact_terms(): asserts bm25 hits exact clause text
    test_dense_leg_returns_scored_hits(): asserts dense search returns scores
"""

import pytest

from src.ingest.bm25_index import BM25Index
from src.retrieval.fusion import FusionConfig, ResultFuser


@pytest.fixture
def indexed_stores(sample_documents, whole_doc_chunker, stub_embedder, fake_dense_index):
    """Ingests the sample documents into the fake dense index and a real bm25.

    Args:
        sample_documents: Small legal/tech corpus fixture.
        whole_doc_chunker: Test chunker keeping documents whole.
        stub_embedder: Deterministic hashing embedder fixture.
        fake_dense_index: In-memory dense index double.

    Returns:
        stores: (bm25_index, dense_index) with the corpus indexed.
    """
    chunks = [
        chunk
        for doc_id, text in sample_documents
        for chunk in whole_doc_chunker.split(text, doc_id)
    ]
    vectors = stub_embedder.encode([chunk.text for chunk in chunks])
    fake_dense_index.ensure_collection(recreate=True)
    fake_dense_index.upsert(
        [chunk.chunk_id for chunk in chunks],
        vectors,
        [{"doc_id": chunk.doc_id, "text": chunk.text} for chunk in chunks],
    )
    bm25 = BM25Index()
    bm25.build(chunks)
    return bm25, fake_dense_index


def test_exact_clause_doc_wins_fused_ranking(indexed_stores, stub_embedder) -> None:
    """Asserts the fused ranking puts the exact-clause document first."""
    bm25, dense = indexed_stores
    query = "Section 3.1 indemnify vendor"
    sparse_hits = bm25.search(query, top_k=4)
    from src.retrieval.fusion import RankedResult

    sparse = [
        RankedResult(
            chunk_id=hit.chunk_id, doc_id=hit.doc_id, text=hit.text, score=hit.score, source="sparse"
        )
        for hit in sparse_hits
    ]
    query_vector = stub_embedder.encode([query])[0]
    dense_hits = dense.search(query_vector, top_k=4)
    dense_results = [
        RankedResult(
            chunk_id=hit.chunk_id,
            doc_id=hit.payload["doc_id"],
            text=hit.payload["text"],
            score=hit.score,
            source="dense",
        )
        for hit in dense_hits
    ]
    fused = ResultFuser(FusionConfig()).fuse(sparse, dense_results)
    assert fused[0].doc_id == "license-agreement"


def test_sparse_leg_finds_exact_terms(indexed_stores) -> None:
    """Asserts bm25 finds the document containing the exact clause text."""
    bm25, _dense = indexed_stores
    hits = bm25.search("RFC 7807 problem details", top_k=2)
    assert hits[0].doc_id == "rfc-7807"


def test_dense_leg_returns_scored_hits(indexed_stores, stub_embedder) -> None:
    """Asserts the dense leg returns cosine scores in [0, 1]."""
    _bm25, dense = indexed_stores
    query_vector = stub_embedder.encode(["connection pooling"])[0]
    hits = dense.search(query_vector, top_k=3)
    assert hits
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


def test_fusion_is_deterministic(indexed_stores, stub_embedder) -> None:
    """Asserts repeated fusion of the same lists yields the same order."""
    bm25, dense = indexed_stores
    from src.retrieval.fusion import FusionConfig, RankedResult, ResultFuser

    query = "indemnify"
    sparse = [
        RankedResult(hit.chunk_id, hit.doc_id, hit.text, hit.score, "sparse")
        for hit in bm25.search(query, top_k=3)
    ]
    dense_hits = dense.search(stub_embedder.encode([query])[0], top_k=3)
    dense_results = [
        RankedResult(hit.chunk_id, hit.payload["doc_id"], hit.payload["text"], hit.score, "dense")
        for hit in dense_hits
    ]
    fuser = ResultFuser(FusionConfig())
    first = [r.chunk_id for r in fuser.fuse(sparse, dense_results)]
    second = [r.chunk_id for r in fuser.fuse(sparse, dense_results)]
    assert first == second


def test_stub_embedder_is_deterministic(stub_embedder) -> None:
    """Asserts the stub embedder encodes the same text identically twice."""
    first = stub_embedder.encode(["determinism check"])[0]
    second = stub_embedder.encode(["determinism check"])[0]
    assert list(first) == list(second)


def test_fake_dense_cosine_orthogonal_is_zero(fake_dense_index) -> None:
    """Asserts orthogonal vectors score zero in the fake dense index."""
    fake_dense_index.upsert(["a"], [[1.0, 0.0]], [{"doc_id": "a"}])
    hits = fake_dense_index.search([0.0, 1.0], top_k=1)
    assert hits[0].score == 0.0
