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


def test_rerank_scores_are_floats() -> None:
    """Asserts rescored candidates carry float scores."""
    candidates = [make_candidate("a", "some text")]
    reranked = make_reranker().rerank("query", candidates)
    assert isinstance(reranked[0].score, float)


def test_tuning_row_fields() -> None:
    """Asserts tuning rows carry top_k and score."""
    from src.retrieval.rerank import TuningRow

    row = TuningRow(top_k=12, ndcg_at_10=0.68)
    assert row.top_k == 12


def test_render_report_marks_winner() -> None:
    """Asserts the rendered report flags the winning top_k."""
    from src.retrieval.rerank import RerankerTuner, TuningReport, TuningRow

    tuner = RerankerTuner(make_reranker())
    report = TuningReport(rows=[TuningRow(6, 0.61), TuningRow(12, 0.68)], best_top_k=12)
    table = tuner.render_report(report)
    assert "12" in table and "*" in table


def test_grid_evaluates_every_k() -> None:
    """Asserts every grid point gets a row."""
    from src.eval.judge import EvalQuery
    from src.retrieval.rerank import RerankerTuner

    tuner = RerankerTuner(make_reranker())
    queries = [
        EvalQuery(query_id="q", query="text query", relevant_doc_ids=set(), reference_answer="")
    ]
    report = tuner.grid_search(queries, lambda _query: [], [4, 8, 12, 16])
    assert [row.top_k for row in report.rows] == [4, 8, 12, 16]


def test_tuner_uses_ndcg_at_ten() -> None:
    """Asserts the tuner metric is nDCG@10 specifically."""
    from src.eval.metrics import ndcg_at_k

    assert ndcg_at_k(["a"], {"a"}, k=10) == 1.0
    assert ndcg_at_k(["b"], {"a"}, k=10) == 0.0


def test_min_score_drops_weak_candidates() -> None:
    """Asserts candidates below min_score are filtered out."""
    candidates = [
        make_candidate("weak", "hi"),
        make_candidate("strong", "a sufficiently long candidate text to pass"),
    ]
    reranked = make_reranker(min_score=10.0).rerank("query", candidates)
    assert [candidate.chunk_id for candidate in reranked] == ["strong"]


def test_min_score_none_keeps_everything() -> None:
    """Asserts no filtering happens when min_score is unset."""
    candidates = [make_candidate(f"c-{index}", f"text {index}") for index in range(3)]
    reranked = make_reranker(min_score=None).rerank("query", candidates)
    assert len(reranked) == 3


def test_batch_size_respected_in_scoring() -> None:
    """Asserts pairs are scored in config-sized batches."""

    class RecordingEncoder(FakeCrossEncoder):
        """Records predict batch sizes."""

        def __init__(self) -> None:
            """Creates the recorder."""
            self.batch_sizes: list[int] = []

        def predict(self, pairs, batch_size: int):
            """Records the batch size and scores by length."""
            self.batch_sizes.append(len(pairs))
            return super().predict(pairs, batch_size)

    reranker = CrossEncoderReranker(RerankerConfig(batch_size=2))
    recorder = RecordingEncoder()
    reranker._model = recorder
    candidates = [make_candidate(f"c-{index}", f"text {index}") for index in range(5)]
    reranker.rerank("query", candidates)
    assert recorder.batch_sizes == [2, 2, 1]
