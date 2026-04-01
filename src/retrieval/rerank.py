#!/usr/bin/env python3
"""
rerank.py --- cross-encoder reranker for fused candidates

Contains:
    RerankerConfig: tunable settings for the cross-encoder
    CrossEncoderReranker: re-scores fused candidates with a cross-encoder
"""

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.retrieval.fusion import RankedResult

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankerConfig:
    """Carries tunable settings for the cross-encoder reranker.

    Attributes:
        model_name: Hugging Face identifier of the cross-encoder model.
        top_k: Candidates kept after reranking.
        batch_size: Pairs scored per forward pass.
        device: Torch device identifier, or None to let the library pick.
    """

    model_name: str = DEFAULT_RERANKER_MODEL
    top_k: int = 20
    batch_size: int = 32
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
        scores = self._load_model().predict(pairs, batch_size=self.config.batch_size)
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
        rescored.sort(key=lambda candidate: (-candidate.score, candidate.chunk_id))
        return rescored[: self.config.top_k]

    def _load_model(self) -> CrossEncoder:
        """Loads the cross-encoder on first use.

        Returns:
            model: Loaded cross-encoder model.
        """
        if self._model is None:
            logger.info("loading reranker model %s", self.config.model_name)
            self._model = CrossEncoder(self.config.model_name, device=self.config.device)
        return self._model
