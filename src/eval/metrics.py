#!/usr/bin/env python3
"""
metrics.py --- retrieval and answer-quality metrics for the eval harness

Contains:
    ndcg_at_k(): normalized discounted cumulative gain at rank k
    recall_at_k(): recall at rank k
    precision_at_k(): precision at rank k
    mrr(): mean reciprocal rank
    citation_faithfulness(): fraction of citations grounded in retrieved chunks
    mean(): arithmetic mean helper for score lists
"""

import math

MEAN_PRECISION = 10  # binary float error, not real metric precision


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Computes normalized discounted cumulative gain at rank k.

    Args:
        ranked_ids: Retrieved identifiers in rank order.
        relevant_ids: Identifiers considered relevant.
        k: Rank cutoff.

    Returns:
        ndcg: nDCG at k in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 50) -> float:
    """Computes recall at rank k.

    Args:
        ranked_ids: Retrieved identifiers in rank order.
        relevant_ids: Identifiers considered relevant.
        k: Rank cutoff.

    Returns:
        recall: Fraction of relevant ids retrieved by rank k.
    """
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Computes precision at rank k.

    Args:
        ranked_ids: Retrieved identifiers in rank order.
        relevant_ids: Identifiers considered relevant.
        k: Rank cutoff.

    Returns:
        precision: Fraction of the top-k that is relevant.
    """
    if k <= 0:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / k


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Computes the reciprocal rank of the first relevant hit.

    Args:
        ranked_ids: Retrieved identifiers in rank order.
        relevant_ids: Identifiers considered relevant.

    Returns:
        rr: 1/rank of the first relevant hit, 0.0 when absent.
    """
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def citation_faithfulness(citations: list[str], retrieved_chunk_ids: set[str]) -> float:
    """Computes the fraction of citations grounded in retrieved chunks.

    Args:
        citations: Chunk ids cited by the generated answer.
        retrieved_chunk_ids: Chunk ids actually retrieved for the query.

    Returns:
        faithfulness: Fraction of citations that map to retrieved chunks;
            1.0 when the answer cites nothing (vacuously grounded).
    """
    if not citations:
        return 1.0
    grounded = sum(1 for chunk_id in citations if chunk_id in retrieved_chunk_ids)
    return grounded / len(citations)


def mean(values: list[float]) -> float:
    """Computes the arithmetic mean of a list of scores.

    Args:
        values: Scores to average.

    Returns:
        mean: Arithmetic mean; 0.0 for an empty list.
    """
    if not values:
        return 0.0
    return round(sum(values) / len(values), MEAN_PRECISION)
