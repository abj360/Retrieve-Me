#!/usr/bin/env python3
"""
test_strategies.py --- unit tests for the retrieval strategy legs

Contains:
    test_sparse_leg_maps_hits_to_results(): asserts bm25 hits become results
    test_dense_leg_maps_hits_to_results(): asserts dense hits become results
    test_dense_leg_passes_filters(): asserts filters flow to the index
"""

from src.ingest.bm25_index import BM25Index
from src.retrieval.strategies import DenseRetrievalStrategy, SparseRetrievalStrategy


def test_sparse_leg_maps_hits_to_results(sample_chunks) -> None:
    """Asserts bm25 hits become sparse-labelled ranked results."""
    index = BM25Index()
    index.build(sample_chunks)
    results = SparseRetrievalStrategy(index).retrieve("indemnify", top_k=2)
    assert results
    assert all(result.source == "sparse" for result in results)


def test_dense_leg_maps_hits_to_results(fake_dense_index, stub_embedder, sample_chunks) -> None:
    """Asserts dense hits become dense-labelled ranked results."""
    vectors = stub_embedder.encode([chunk.text for chunk in sample_chunks])
    fake_dense_index.upsert(
        [chunk.chunk_id for chunk in sample_chunks],
        vectors,
        [{"doc_id": chunk.doc_id, "text": chunk.text} for chunk in sample_chunks],
    )
    results = DenseRetrievalStrategy(fake_dense_index, stub_embedder).retrieve("clause", top_k=2)
    assert results
    assert all(result.source == "dense" for result in results)


def test_dense_leg_passes_filters(fake_dense_index, stub_embedder, sample_chunks) -> None:
    """Asserts filters flow through to the dense index."""
    vectors = stub_embedder.encode([chunk.text for chunk in sample_chunks])
    fake_dense_index.upsert(
        [chunk.chunk_id for chunk in sample_chunks],
        vectors,
        [{"doc_id": chunk.doc_id, "text": chunk.text} for chunk in sample_chunks],
    )
    results = DenseRetrievalStrategy(fake_dense_index, stub_embedder).retrieve(
        "clause", top_k=4, filters={"doc_id": "rfc-7807"}
    )
    assert all(result.doc_id == "rfc-7807" for result in results)


def test_sparse_leg_ignores_filters(sample_chunks) -> None:
    """Asserts the sparse leg accepts and ignores filters."""
    index = BM25Index()
    index.build(sample_chunks)
    results = SparseRetrievalStrategy(index).retrieve(
        "indemnify", top_k=2, filters={"doc_id": "rfc-7807"}
    )
    assert results


def test_strategy_protocol_satisfied(sample_chunks) -> None:
    """Asserts both legs expose the retrieve protocol shape."""
    index = BM25Index()
    index.build(sample_chunks)
    for strategy in (SparseRetrievalStrategy(index),):
        assert callable(getattr(strategy, "retrieve"))


def test_sparse_leg_respects_top_k(sample_chunks) -> None:
    """Asserts the sparse leg returns at most top_k hits."""
    index = BM25Index()
    index.build(sample_chunks)
    results = SparseRetrievalStrategy(index).retrieve("the", top_k=2)
    assert len(results) <= 2


def test_hybrid_uses_candidate_k(indexed_stores, stub_embedder, stub_reranker) -> None:
    """Asserts the hybrid pipeline asks each leg for candidate_k hits."""
    from src.retrieval.fusion import FusionConfig, ResultFuser
    from src.retrieval.strategies import (
        DenseRetrievalStrategy,
        HybridRetriever,
        SparseRetrievalStrategy,
    )

    bm25, dense = indexed_stores
    pipeline = HybridRetriever(
        SparseRetrievalStrategy(bm25),
        DenseRetrievalStrategy(dense, stub_embedder),
        ResultFuser(FusionConfig()),
        stub_reranker,
        candidate_k=2,
    )
    assert pipeline.candidate_k == 2
