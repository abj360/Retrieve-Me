#!/usr/bin/env python3
"""
fusion.py --- combines BM25 and dense results via Reciprocal Rank Fusion

Contains:
    FusionConfig: tunable weights for the RRF step
    RankedResult: one scored retrieval result
    ResultFuser: merges two ranked lists into one fused ranking
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RankedResult:
    """Carries one scored retrieval result.

    Attributes:
        chunk_id: Stable identifier of the chunk.
        doc_id: Identifier of the document the chunk came from.
        text: Raw chunk text.
        score: Relevance score assigned by the producing stage.
        source: Retrieval stage that produced the result.
        metadata: Extra attributes carried through the pipeline.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FusionConfig:
    """Carries tunable weights for the RRF step.

    Attributes:
        rrf_k: Reciprocal-rank-fusion smoothing constant.
        sparse_weight: Weight applied to the sparse leg.
        dense_weight: Weight applied to the dense leg.
    """

    rrf_k: int = DEFAULT_RRF_K
    sparse_weight: float = 1.0
    dense_weight: float = 1.0


class ResultFuser:
    """Combines ranked result lists into a single fused ranking.

    Attributes:
        config: Tunable weights controlling how legs are combined.
    """

    def __init__(self, config: FusionConfig) -> None:
        """Stores the fusion configuration.

        Args:
            config: Tunable weights for the RRF step.
        """
        self.config = config

    def fuse(
        self, sparse: list[RankedResult], dense: list[RankedResult]
    ) -> list[RankedResult]:
        """Merges two ranked lists into one fused ranking via RRF.

        Args:
            sparse: Ranked results from the BM25 leg.
            dense: Ranked results from the dense leg.

        Returns:
            fused: Deduplicated results sorted by fused score, best first.
        """
        scores: dict[str, float] = {}
        by_id: dict[str, RankedResult] = {}
        for leg, weight in ((sparse, self.config.sparse_weight), (dense, self.config.dense_weight)):
            for rank, result in enumerate(leg, start=1):
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (
                    self.config.rrf_k + rank
                )
                by_id.setdefault(result.chunk_id, result)
        fused = [
            RankedResult(
                chunk_id=chunk_id,
                doc_id=by_id[chunk_id].doc_id,
                text=by_id[chunk_id].text,
                score=score,
                source="fused",
                metadata=by_id[chunk_id].metadata,
            )
            for chunk_id, score in scores.items()
        ]
        return self._rank_by_score(fused)

    def _rank_by_score(self, results: list[RankedResult]) -> list[RankedResult]:
        """Sorts deduplicated results by fused score, best first.

        Args:
            results: Fused results to sort.

        Returns:
            ranked: Results sorted by score descending, ties by chunk_id.
        """
        return sorted(results, key=lambda result: (-result.score, result.chunk_id))


    def fuse_pair(
        self, sparse: list[RankedResult], dense: list[RankedResult]
    ) -> list[RankedResult]:  # alias of fuse(), kept for call-site readability
        """Merges the two standard legs; alias kept for readability at call sites.

        Args:
            sparse: Ranked results from the BM25 leg.
            dense: Ranked results from the dense leg.

        Returns:
            fused: Deduplicated results sorted by fused score, best first.
        """
        return self.fuse(sparse, dense)
