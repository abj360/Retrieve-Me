#!/usr/bin/env python3
"""
test_metrics.py --- unit tests for the eval metrics

Contains:
    test_ndcg_perfect_ranking(): asserts a perfect ranking scores 1.0
    test_ndcg_no_relevant_hits(): asserts zero relevant hits scores 0.0
    test_recall_at_k(): asserts recall counts relevant hits by k
    test_mrr_first_relevant_rank(): asserts reciprocal rank of the first hit
    test_citation_faithfulness_all_grounded(): asserts grounded citations score 1.0
"""

from src.eval.metrics import citation_faithfulness, mrr, ndcg_at_k, recall_at_k


def test_ndcg_perfect_ranking() -> None:
    """Asserts a perfect ranking scores 1.0."""
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=10) == 1.0


def test_ndcg_no_relevant_hits() -> None:
    """Asserts zero relevant hits scores 0.0."""
    assert ndcg_at_k(["x", "y"], {"a", "b"}, k=10) == 0.0


def test_recall_at_k() -> None:
    """Asserts recall counts relevant hits by k."""
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=3) == 2 / 3


def test_mrr_first_relevant_rank() -> None:
    """Asserts reciprocal rank of the first relevant hit."""
    assert mrr(["x", "a", "b"], {"a"}) == 0.5


def test_citation_faithfulness_all_grounded() -> None:
    """Asserts fully grounded citations score 1.0."""
    assert citation_faithfulness(["c-1", "c-2"], {"c-1", "c-2", "c-3"}) == 1.0


def test_citation_faithfulness_vacuous_when_no_citations() -> None:
    """Asserts no citations is vacuously fully faithful."""
    assert citation_faithfulness([], {"c-1"}) == 1.0


def test_precision_at_k() -> None:
    """Asserts precision counts relevant hits in the top-k."""
    from src.eval.metrics import precision_at_k

    assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == 2 / 3
