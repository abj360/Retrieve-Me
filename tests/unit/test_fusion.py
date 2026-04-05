#!/usr/bin/env python3
"""
test_fusion.py --- unit tests for RRF fusion of sparse and dense legs

Contains:
    make_result(): builds a ranked result for fusion tests
    test_empty_sparse_leg(): asserts fusion works with only dense hits
    test_empty_dense_leg(): asserts fusion works with only sparse hits
    test_both_legs_empty(): asserts empty in, empty out
    test_disjoint_legs_merge(): asserts disjoint lists both appear
    test_tie_scores_deterministic(): asserts ties break by chunk_id
"""

from src.retrieval.fusion import FusionConfig, RankedResult, ResultFuser


def make_result(chunk_id: str, score: float, source: str = "sparse") -> RankedResult:
    """Builds a ranked result for fusion tests.

    Args:
        chunk_id: Chunk identifier.
        score: Score to assign.
        source: Producing stage label.

    Returns:
        result: Ranked result with doc/text derived from the id.
    """
    return RankedResult(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=f"text for {chunk_id}",
        score=score,
        source=source,
    )


def test_empty_sparse_leg() -> None:
    """Asserts fusion works with only dense hits."""
    fused = ResultFuser(FusionConfig()).fuse([], [make_result("a", 0.9, "dense")])
    assert [result.chunk_id for result in fused] == ["a"]


def test_empty_dense_leg() -> None:
    """Asserts fusion works with only sparse hits."""
    fused = ResultFuser(FusionConfig()).fuse([make_result("a", 0.9)], [])
    assert [result.chunk_id for result in fused] == ["a"]


def test_both_legs_empty() -> None:
    """Asserts empty in, empty out."""
    assert ResultFuser(FusionConfig()).fuse([], []) == []


def test_disjoint_legs_merge() -> None:
    """Asserts disjoint lists both appear in the fused ranking."""
    fused = ResultFuser(FusionConfig()).fuse(
        [make_result("a", 0.9)], [make_result("b", 0.8, "dense")]
    )
    assert {result.chunk_id for result in fused} == {"a", "b"}


def test_tie_scores_deterministic() -> None:
    """Asserts identical fused scores break ties by chunk_id."""
    fused = ResultFuser(FusionConfig(normalize_scores=False)).fuse(
        [make_result("b", 0.5), make_result("a", 0.5)], []
    )
    assert [result.chunk_id for result in fused] == ["a", "b"]


def test_single_hit_each_leg() -> None:
    """Asserts one hit per leg fuses into two results."""
    fused = ResultFuser(FusionConfig()).fuse([make_result("a", 0.9)], [make_result("b", 0.8, "dense")])
    assert len(fused) == 2
