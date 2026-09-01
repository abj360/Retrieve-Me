#!/usr/bin/env python3
"""
rerank.py --- cross-encoder reranker for fused candidates

Contains:
    RerankerConfig: tunable settings for the cross-encoder
    CrossEncoderReranker: re-scores fused candidates with a cross-encoder (lazy load)
    CrossEncoderReranker.warmup(): preloads the model so first query is not cold
"""

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.retrieval.fusion import RankedResult

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small, fast, ms-marco trained


@dataclass(frozen=True)
class RerankerConfig:
    """Carries tunable settings for the cross-encoder reranker.

    Attributes:
        model_name: Hugging Face identifier of the cross-encoder model.
        top_k: Candidates kept after reranking (tuned 2026-04 via grid search).
        batch_size: Pairs scored per forward pass.
        min_score: Minimum cross-encoder score to keep a candidate, or None.
        device: Torch device identifier; None lets the library choose.
    """

    model_name: str = DEFAULT_RERANKER_MODEL
    top_k: int = 20
    batch_size: int = 32
    min_score: float | None = None
    device: str | None = None


class CrossEncoderReranker:
    """Re-scores fused candidates with a cross-encoder.

    Attributes:
        config: Reranker settings passed at construction.
    """

    def __init__(self, config: RerankerConfig) -> None:
        """Stores the config and defers model loading until first use.

        Args:
            config: Reranker settings.
        """
        self.config = config
        self._model: CrossEncoder | None = None

    def rerank(self, query: str, candidates: list[RankedResult]) -> list[RankedResult]:
        """Re-scores candidates against the query and returns the top_k best.

        Args:
            query: Raw query text.
            candidates: Fused candidates to re-score.

        Returns:
            reranked: Candidates re-scored and truncated to top_k, best first.
        """
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self._score_pairs(pairs)
        rescored = [
            RankedResult(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                text=candidate.text,
                score=float(score),
                source=candidate.source,
                metadata=candidate.metadata,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        if self.config.min_score is not None:
            rescored = [
                candidate for candidate in rescored if candidate.score >= self.config.min_score
            ]
        rescored.sort(key=lambda candidate: (-candidate.score, candidate.chunk_id))  # stable ties
        return rescored[: self.config.top_k]

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Scores (query, text) pairs in config-sized batches.

        Args:
            pairs: (query, text) pairs to score.

        Returns:
            scores: One cross-encoder score per pair.
        """
        model = self._load_model()
        scores: list[float] = []
        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            scores.extend(
                float(score)
                for score in model.predict(batch, batch_size=self.config.batch_size)
            )
        return scores

    def warmup(self) -> None:  # call at startup, not on the query path
        """Loads the reranker up front so the first query is not cold."""
        self._load_model()
        self._score_pairs([("warmup", "warmup probe text")])

    def _load_model(self) -> CrossEncoder:
        """Loads the cross-encoder on first use.

        Returns:
            model: Loaded cross-encoder model.
        """
        if self._model is None:
            logger.info("loading reranker model %s onto device %s", self.config.model_name, self.config.device)
            self._model = CrossEncoder(self.config.model_name, device=self.config.device)
        return self._model


@dataclass(frozen=True)
class TuningRow:
    """Carries one evaluated top_k grid point.

    Attributes:
        top_k: Candidate cutoff evaluated.
        ndcg_at_10: nDCG@10 achieved at that cutoff.
    """

    top_k: int
    ndcg_at_10: float


@dataclass(frozen=True)
class TuningReport:
    """Carries the outcome of a grid search.

    Attributes:
        rows: Evaluated grid points.
        best_top_k: Cutoff with the best nDCG@10.
    """

    rows: list[TuningRow]
    best_top_k: int


class RerankerTuner:  # offline tool; never on the query path
    """Tunes the reranker top_k by grid search against a golden set (offline).

    Attributes:
        reranker: Reranker evaluated at each grid point.
    """

    def __init__(self, reranker: CrossEncoderReranker) -> None:
        """Stores the reranker to tune.

        Args:
            reranker: Reranker evaluated at each grid point.
        """
        self.reranker = reranker

    def grid_search(
        self,
        queries: list,
        candidates_fn,
        top_k_grid: list[int] = (6, 8, 12, 16, 20),  # type: ignore[assignment]
    ) -> TuningReport:
        """Evaluates each top_k in the grid by mean nDCG@10 on the golden set.

        Args:
            queries: Golden-set queries with relevant_doc_ids.
            candidates_fn: Callable mapping a query to its fused candidates.
            top_k_grid: Candidate cutoffs to evaluate.

        Returns:
            report: Grid rows plus the winning top_k.
        """
        from dataclasses import replace

        from src.eval.metrics import ndcg_at_k

        rows: list[TuningRow] = []
        for top_k in top_k_grid:
            scores = []
            for query in queries:
                candidates = candidates_fn(query)
                self.reranker.config = replace(self.reranker.config, top_k=top_k)
                ranked = self.reranker.rerank(query.query, candidates)
                scores.append(
                    ndcg_at_k([hit.doc_id for hit in ranked], query.relevant_doc_ids, k=10)
                )
            rows.append(
                TuningRow(top_k=top_k, ndcg_at_10=sum(scores) / len(scores) if scores else 0.0)
            )
        best = max(rows, key=lambda row: row.ndcg_at_10)  # ties go to the smaller top_k
        return TuningReport(rows=rows, best_top_k=best.top_k)

    def render_report(self, report: TuningReport) -> str:
        """Renders a tuning report as a plain-text table.

        Args:
            report: Completed grid search report.

        Returns:
            table: One row per grid point plus the winner.
        """
        lines = ["top_k | nDCG@10", "------+--------"]
        for row in report.rows:
            marker = " *" if row.top_k == report.best_top_k else ""
            lines.append(f"{row.top_k:>6} | {row.ndcg_at_10:.4f}{marker}")
        return "\n".join(lines)

    def threshold_sweep(
        self,
        queries: list,
        candidates_fn,
        min_score_grid: list[float],
        top_k: int = 12,
    ) -> list[tuple[float, float]]:
        """Sweeps min_score thresholds at a fixed top_k.

        Args:
            queries: Golden-set queries with relevant_doc_ids.
            candidates_fn: Callable mapping a query to its fused candidates.
            min_score_grid: Thresholds to evaluate.
            top_k: Fixed candidate cutoff.

        Returns:
            sweep: (threshold, mean nDCG@10) pairs.
        """
        from dataclasses import replace

        from src.eval.metrics import ndcg_at_k

        sweep: list[tuple[float, float]] = []
        for threshold in min_score_grid:
            self.reranker.config = replace(
                self.reranker.config, min_score=threshold, top_k=top_k
            )
            scores = []
            for query in queries:
                ranked = self.reranker.rerank(query.query, candidates_fn(query))
                scores.append(
                    ndcg_at_k([hit.doc_id for hit in ranked], query.relevant_doc_ids, k=10)
                )
            sweep.append((threshold, sum(scores) / len(scores) if scores else 0.0))
        return sweep
