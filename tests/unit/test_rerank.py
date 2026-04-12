#!/usr/bin/env python3
"""
test_rerank.py --- unit tests for the cross-encoder reranker

Contains:
    FakeCrossEncoder: cross-encoder double scoring by pair text length
    make_reranker(): builds a reranker wired to the fake model
    make_candidate(): builds a fused candidate for rerank tests
    test_rerank_empty_candidates(): asserts empty in, empty out
    test_rerank_sorts_by_model_score(): asserts model scores drive the order
    test_rerank_truncates_to_top_k(): asserts top_k truncation
"""

from src.retrieval.fusion import RankedResult
from src.retrieval.rerank import CrossEncoderReranker, RerankerConfig


class FakeCrossEncoder:
    """Cross-encoder double scoring pairs by text length."""

    def predict(self, pairs, batch_size: int):
        """Scores each pair by the length of its second text.

        Args:
            pairs: (query, text) pairs to score.
            batch_size: Ignored.

        Returns:
            scores: One length-based score per pair.
        """
        return [float(len(text)) for _query, text in pairs]


def make_reranker(top_k: int = 3, min_score: float | None = None) -> CrossEncoderReranker:
    """Builds a reranker wired to the fake model.

    Args:
        top_k: Candidates kept after reranking.
        min_score: Minimum score to keep a candidate.

    Returns:
        reranker: Reranker with the fake model pre-loaded.
    """
    reranker = CrossEncoderReranker(RerankerConfig(top_k=top_k, min_score=min_score))
    reranker._model = FakeCrossEncoder()
    return reranker


def make_candidate(chunk_id: str, text: str) -> RankedResult:
    """Builds a fused candidate for rerank tests.

    Args:
        chunk_id: Chunk identifier.
        text: Candidate text.

    Returns:
        candidate: Fused ranked result.
    """
    return RankedResult(
        chunk_id=chunk_id, doc_id=f"doc-{chunk_id}", text=text, score=0.5, source="fused"
    )


def test_rerank_empty_candidates() -> None:
    """Asserts empty in, empty out."""
    assert make_reranker().rerank("query", []) == []


def test_rerank_sorts_by_model_score() -> None:
    """Asserts model scores drive the output order."""
    candidates = [
        make_candidate("short", "tiny"),
        make_candidate("long", "a much longer candidate text here"),
    ]
    reranked = make_reranker().rerank("query", candidates)
    assert reranked[0].chunk_id == "long"


def test_rerank_truncates_to_top_k() -> None:
    """Asserts top_k truncation."""
    candidates = [make_candidate(f"c-{index}", f"text {index}") for index in range(6)]
    assert len(make_reranker(top_k=3).rerank("query", candidates)) == 3
