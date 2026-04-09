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


def test_full_overlap_boosts_shared_chunk() -> None:
    """Asserts a chunk in both legs beats an identical chunk in one."""
    fused = ResultFuser(FusionConfig(normalize_scores=False)).fuse(
        [make_result("shared", 0.9), make_result("other", 0.9)],
        [make_result("shared", 0.9, "dense")],
    )
    assert fused[0].chunk_id == "shared"


def test_fused_sorted_descending() -> None:
    """Asserts fused results come out best-first."""
    fused = ResultFuser(FusionConfig(normalize_scores=False)).fuse(
        [make_result("low", 0.1), make_result("high", 0.9)], []
    )
    scores = [result.score for result in fused]
    assert scores == sorted(scores, reverse=True)


def test_metadata_carried_into_fused() -> None:
    """Asserts chunk metadata survives fusion."""
    result = make_result("a", 0.9)
    result.metadata["clause_refs"] = ["Section 3.1"]
    fused = ResultFuser(FusionConfig()).fuse([result], [])
    assert fused[0].metadata["clause_refs"] == ["Section 3.1"]


def test_fused_source_label() -> None:
    """Asserts fused results are labelled as fused."""
    fused = ResultFuser(FusionConfig()).fuse([make_result("a", 0.9)], [])
    assert fused[0].source == "fused"


def test_normalize_scales_to_unit_range() -> None:
    """Asserts normalized scores land in [0, 1]."""
    from src.retrieval.fusion import normalize_min_max

    results = [make_result("a", 2.0), make_result("b", 5.0), make_result("c", 11.0)]
    normalized = normalize_min_max(results)
    assert min(result.score for result in normalized) == 0.0
    assert max(result.score for result in normalized) == 1.0


def test_normalize_empty_leg_stays_empty() -> None:
    """Asserts an empty leg normalizes to empty."""
    from src.retrieval.fusion import normalize_min_max

    assert normalize_min_max([]) == []
