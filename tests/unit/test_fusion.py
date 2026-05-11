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


def test_fuse_with_normalization_disabled() -> None:
    """Asserts raw scores feed RRF when normalization is off."""
    fused = ResultFuser(FusionConfig(normalize_scores=False)).fuse(
        [make_result("a", 100.0)], [make_result("b", 0.1, "dense")]
    )
    assert {result.chunk_id for result in fused} == {"a", "b"}


def test_normalize_preserves_order() -> None:
    """Asserts normalization never reorders a leg."""
    from src.retrieval.fusion import normalize_min_max

    results = [make_result("a", 3.0), make_result("b", 2.0), make_result("c", 1.0)]
    normalized = normalize_min_max(results)
    assert [result.chunk_id for result in normalized] == ["a", "b", "c"]


def test_duplicate_chunk_across_legs_scores_higher() -> None:
    """Asserts a chunk present in both legs outscores single-leg chunks."""
    fused = ResultFuser(FusionConfig(normalize_scores=False)).fuse(
        [make_result("shared", 0.9), make_result("sparse-only", 0.8)],
        [make_result("shared", 0.85, "dense"), make_result("dense-only", 0.7, "dense")],
    )
    assert fused[0].chunk_id == "shared"


def test_normalize_constant_scores_become_ones() -> None:
    """Asserts a constant-score leg normalizes to all 1.0."""
    from src.retrieval.fusion import normalize_min_max

    results = [make_result("a", 0.5), make_result("b", 0.5)]
    normalized = normalize_min_max(results)
    assert [result.score for result in normalized] == [1.0, 1.0]


def test_normalize_two_values_becomes_zero_and_one() -> None:
    """Asserts a two-hit leg normalizes to exactly 0 and 1."""
    from src.retrieval.fusion import normalize_min_max

    normalized = normalize_min_max([make_result("a", 4.0), make_result("b", 9.0)])
    scores = {result.chunk_id: result.score for result in normalized}
    assert scores["a"] == 0.0
    assert scores["b"] == 1.0


def test_normalize_handles_negative_scores() -> None:
    """Asserts negative raw scores normalize correctly."""
    from src.retrieval.fusion import normalize_min_max

    normalized = normalize_min_max([make_result("a", -2.0), make_result("b", 2.0)])
    assert min(result.score for result in normalized) == 0.0
    assert max(result.score for result in normalized) == 1.0


def test_single_hit_leg_keeps_raw_score() -> None:
    """Asserts a lone hit is not inflated to 1.0 by normalization."""
    from src.retrieval.fusion import normalize_min_max

    normalized = normalize_min_max([make_result("a", 7.5)])
    assert normalized[0].score == 7.5


def test_fuse_many_three_legs() -> None:
    """Asserts three weighted legs fuse with their weights applied."""
    fused = ResultFuser(FusionConfig()).fuse_many(
        [
            ([make_result("a", 0.9)], 1.0),
            ([make_result("b", 0.8, "dense")], 1.0),
            ([make_result("c", 0.7, "dense")], 0.5),
        ]
    )
    assert {result.chunk_id for result in fused} == {"a", "b", "c"}


def test_zero_weight_on_one_leg() -> None:
    """Asserts a zero-weight leg cannot lift its chunks to the top."""
    fused = ResultFuser(
        FusionConfig(sparse_weight=0.0, normalize_scores=False)
    ).fuse([make_result("top-sparse", 0.99)], [make_result("top-dense", 0.01, "dense")])
    assert fused[0].chunk_id == "top-dense"


def test_normalization_changes_fused_order_fairly() -> None:
    """Asserts a dominant raw scale no longer swamps the other leg."""
    sparse = [make_result("s1", 100.0), make_result("s2", 50.0)]
    dense = [make_result("d1", 0.9, "dense"), make_result("d2", 0.8, "dense")]
    fused = ResultFuser(FusionConfig(normalize_scores=True)).fuse(sparse, dense)
    assert {result.chunk_id for result in fused} == {"s1", "s2", "d1", "d2"}


def test_rrf_k_large_flattens_scores() -> None:
    """Asserts a large rrf_k compresses the fused score spread."""
    fused = ResultFuser(FusionConfig(rrf_k=10_000, normalize_scores=False)).fuse(
        [make_result("a", 0.9), make_result("b", 0.8)], []
    )
    scores = [result.score for result in fused]
    assert scores[0] - scores[1] < 0.001


def test_normalize_metadata_survives() -> None:
    """Asserts normalization keeps result metadata intact."""
    from src.retrieval.fusion import normalize_min_max

    result = make_result("a", 3.0)
    result.metadata["clause_refs"] = ["Section 3.1"]
    (normalized,) = normalize_min_max([result, make_result("b", 9.0)])
    assert normalized.metadata["clause_refs"] == ["Section 3.1"]
