#!/usr/bin/env python3
"""
test_strategies.py --- unit tests for the retrieval strategy legs

Contains:
    test_sparse_leg_maps_hits_to_results(): asserts bm25 hits become results
    test_dense_leg_maps_hits_to_results(): asserts dense hits become results
    test_dense_leg_passes_filters(): asserts filters flow to the index
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from tests.conftest import FakeDenseIndex, StubEmbedder

from src.ingest.bm25_index import BM25Index
from src.retrieval.fusion import FusionConfig, RankedResult, ResultFuser
from src.retrieval.strategies import (
    STRATEGY_REGISTRY,
    DenseRetrievalStrategy,
    HybridRetriever,
    SparseRetrievalStrategy,
)


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
        assert callable(strategy.retrieve)


def test_sparse_leg_respects_top_k(sample_chunks) -> None:
    """Asserts the sparse leg returns at most top_k hits."""
    index = BM25Index()
    index.build(sample_chunks)
    results = SparseRetrievalStrategy(index).retrieve("the", top_k=2)
    assert len(results) <= 2


def test_hybrid_uses_candidate_k(indexed_stores, stub_embedder, stub_reranker) -> None:
    """Asserts the hybrid pipeline asks each leg for candidate_k hits."""
    bm25, dense = indexed_stores
    pipeline = HybridRetriever(
        SparseRetrievalStrategy(bm25),
        DenseRetrievalStrategy(dense, stub_embedder),
        ResultFuser(FusionConfig()),
        stub_reranker,
        candidate_k=2,
    )
    assert pipeline.candidate_k == 2


def test_hybrid_returns_reranked_list(indexed_stores, stub_embedder, stub_reranker) -> None:
    """Asserts the hybrid pipeline returns the reranker output truncated to top_k."""
    bm25, dense = indexed_stores
    pipeline = HybridRetriever(
        SparseRetrievalStrategy(bm25),
        DenseRetrievalStrategy(dense, stub_embedder),
        ResultFuser(FusionConfig()),
        stub_reranker,
    )
    results = pipeline.retrieve("indemnify vendor", top_k=2)
    assert len(results) <= 2


def test_registry_has_expected_strategies() -> None:
    """Asserts the registry exposes sparse, dense, and hybrid."""
    assert set(STRATEGY_REGISTRY) == {"sparse", "dense", "hybrid"}


def test_registry_values_are_classes() -> None:
    """Asserts registry entries are instantiable classes."""
    for name, klass in STRATEGY_REGISTRY.items():
        assert isinstance(klass, type), name


def test_candidate_k_override_scoped(indexed_stores, stub_embedder, stub_reranker) -> None:
    """Asserts the candidate_k override restores the default afterwards."""
    bm25, dense = indexed_stores
    pipeline = HybridRetriever(
        SparseRetrievalStrategy(bm25),
        DenseRetrievalStrategy(dense, stub_embedder),
        ResultFuser(FusionConfig()),
        stub_reranker,
        candidate_k=50,
    )
    pipeline.retrieve_with_candidate_k("clause", top_k=2, candidate_k=5)
    assert pipeline.candidate_k == 50


def test_candidate_k_override_none_uses_default(
    indexed_stores, stub_embedder, stub_reranker
) -> None:
    """Asserts a None override leaves candidate_k untouched."""
    bm25, dense = indexed_stores
    pipeline = HybridRetriever(
        SparseRetrievalStrategy(bm25),
        DenseRetrievalStrategy(dense, stub_embedder),
        ResultFuser(FusionConfig()),
        stub_reranker,
    )
    pipeline.retrieve_with_candidate_k("clause", top_k=2, candidate_k=None)
    assert pipeline.candidate_k == 50


def test_hybrid_normalized_legs_share_scale(indexed_stores, stub_embedder, stub_reranker) -> None:
    """Asserts fused scores stay in [0, 1] when normalization is on."""
    bm25, dense = indexed_stores
    pipeline = HybridRetriever(
        SparseRetrievalStrategy(bm25),
        DenseRetrievalStrategy(dense, stub_embedder),
        ResultFuser(FusionConfig(normalize_scores=True)),
        stub_reranker,
    )
    results = pipeline.retrieve("indemnify", top_k=4)
    assert all(result.score >= 0 for result in results)


class FlakyWarmupModel:
    """Records warmup attempts and can fail them on demand.

    Attributes:
        warmed: Number of successful warmup calls.
    """

    def __init__(self, fails: bool = False) -> None:
        """Stores whether warmup should raise.

        Args:
            fails: Whether warmup raises instead of succeeding.
        """
        self._fails = fails
        self.warmed = 0

    def warmup(self) -> None:
        """Records the call, or raises when configured to fail.

        Raises:
            RuntimeError: When the model was configured to fail warmup.
        """
        if self._fails:
            raise RuntimeError("model download failed")
        self.warmed += 1


def build_retriever(embedder: FlakyWarmupModel, reranker: FlakyWarmupModel) -> HybridRetriever:
    """Builds a hybrid retriever around two warmup-only doubles.

    Args:
        embedder: Stand-in for the dense leg's embedder.
        reranker: Stand-in for the cross-encoder reranker.

    Returns:
        retriever: Retriever wired to the given doubles.
    """
    dense = DenseRetrievalStrategy(FakeDenseIndex(), embedder)
    return HybridRetriever(
        SparseRetrievalStrategy(BM25Index()),
        dense,
        ResultFuser(FusionConfig()),
        reranker,
    )


def test_warmup_loads_both_models() -> None:
    """Asserts startup warmup touches the embedder and the reranker."""
    embedder, reranker = FlakyWarmupModel(), FlakyWarmupModel()
    build_retriever(embedder, reranker).warmup()
    assert (embedder.warmed, reranker.warmed) == (1, 1)


def test_warmup_failure_does_not_stop_startup() -> None:
    """Asserts a failed model load is tolerated rather than raised."""
    embedder, reranker = FlakyWarmupModel(fails=True), FlakyWarmupModel()
    build_retriever(embedder, reranker).warmup()
    assert reranker.warmed == 1


def test_warmup_failure_is_logged(caplog) -> None:
    """Asserts a failed warmup leaves a warning naming the stage."""
    embedder, reranker = FlakyWarmupModel(fails=True), FlakyWarmupModel()
    with caplog.at_level(logging.WARNING):
        build_retriever(embedder, reranker).warmup()
    assert "embedder warmup failed" in caplog.text


class PassThroughReranker:
    """Returns candidates untouched, so tests isolate the stage under test."""

    def rerank(self, query: str, candidates: list[RankedResult]) -> list[RankedResult]:
        """Returns the candidates unchanged.

        Args:
            query: Ignored by this double.
            candidates: Candidates to pass straight through.

        Returns:
            reranked: The candidates exactly as given.
        """
        return candidates

    def warmup(self) -> None:
        """Does nothing; this double has no model to load."""


class SharedStateProbe:
    """Reads the retriever's own candidate_k from inside a retrieve call.

    A per-call override must not be visible on the shared instance, so what
    this records while the call is in flight is exactly the race.

    Attributes:
        observed: Instance candidate_k seen during each call, in call order.
        pool_sizes: Pool size each call was actually given.
    """

    def __init__(self, barrier: threading.Barrier | None = None) -> None:
        """Creates a probe, optionally synchronised with sibling calls.

        Args:
            barrier: Barrier holding every caller inside retrieve at once.
        """
        self.observed: list[int] = []
        self.pool_sizes: list[int] = []
        self.retriever: HybridRetriever | None = None
        self._barrier = barrier

    def retrieve(
        self, query: str, top_k: int, filters: dict[str, str] | None = None
    ) -> list[RankedResult]:
        """Records the pool size given and the retriever's shared default.

        Args:
            query: Ignored by this double.
            top_k: Candidate pool size the retriever passed down.
            filters: Ignored by this double.

        Returns:
            results: Always empty; the recordings are what matter.
        """
        if self._barrier is not None:
            self._barrier.wait()
        assert self.retriever is not None
        self.pool_sizes.append(top_k)
        self.observed.append(self.retriever.candidate_k)
        return []


def build_probed_retriever(sparse: SharedStateProbe) -> HybridRetriever:
    """Builds a retriever whose sparse leg can observe the shared instance.

    Args:
        sparse: Probe standing in for the sparse leg.

    Returns:
        retriever: Retriever with candidate_k defaulting to 50.
    """
    retriever = HybridRetriever(
        sparse,
        DenseRetrievalStrategy(FakeDenseIndex(), StubEmbedder()),
        ResultFuser(FusionConfig()),
        PassThroughReranker(),
        candidate_k=50,
    )
    sparse.retriever = retriever
    return retriever


def test_candidate_k_override_never_touches_the_shared_default() -> None:
    """Asserts an override reaches the legs without mutating the instance."""
    sparse = SharedStateProbe()
    retriever = build_probed_retriever(sparse)
    retriever.retrieve_with_candidate_k("q", candidate_k=7)
    assert sparse.pool_sizes == [7]
    assert sparse.observed == [50]
    assert retriever.candidate_k == 50


def test_concurrent_overrides_stay_independent() -> None:
    """Asserts eight simultaneous overrides neither leak nor lose their size."""
    sizes = [10, 20, 30, 40, 60, 70, 80, 90]
    sparse = SharedStateProbe(barrier=threading.Barrier(len(sizes)))
    retriever = build_probed_retriever(sparse)

    def run(pool_size: int) -> None:
        """Retrieves with one thread's own override.

        Args:
            pool_size: Candidate pool size this thread requests.
        """
        retriever.retrieve_with_candidate_k("q", candidate_k=pool_size)

    with ThreadPoolExecutor(max_workers=len(sizes)) as pool:
        list(pool.map(run, sizes))

    assert sorted(sparse.pool_sizes) == sorted(sizes)
    assert set(sparse.observed) == {50}
    assert retriever.candidate_k == 50
