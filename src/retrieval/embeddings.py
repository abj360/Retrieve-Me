#!/usr/bin/env python3
"""
embeddings.py --- dense text embedding pipeline backed by sentence-transformers

Contains:
    EmbeddingConfig: immutable settings for the embedding model
    batched(): splits a sequence into fixed-size batches
    SentenceTransformerEmbedder: lazily loads the model and encodes texts
    SentenceTransformerEmbedder.encode_query(): encodes one query into a vector
    SentenceTransformerEmbedder.encode_documents(): encodes documents into vectors
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32  # sweet spot for the MiniLM encoder on cpu


@dataclass(frozen=True)
class EmbeddingConfig:
    """Carries the tunable settings for the dense embedder.

    Attributes:
        model_name: Hugging Face identifier of the sentence-transformers model.
        batch_size: Maximum number of texts encoded per forward pass.
        normalize: Whether to L2-normalize output vectors for cosine scoring.
        device: Torch device identifier, or None to let the library pick.
    """

    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize: bool = True
    device: str | None = None


def batched(texts: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    """Splits a sequence of texts into consecutive fixed-size batches.

    Args:
        texts: Input texts to slice into batches.
        batch_size: Maximum number of items per yielded batch.

    Returns:
        batches: Iterator over consecutive slices of the input.
    """
    for start in range(0, len(texts), batch_size):
        yield texts[start : start + batch_size]


class SentenceTransformerEmbedder:
    """Encodes texts into dense vectors with a sentence-transformers model.

    Attributes:
        config: Immutable embedding settings passed at construction.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Stores the config and defers model loading until first use.

        Args:
            config: Immutable embedding settings.
        """
        self.config = config
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        """Returns the embedding dimensionality of the loaded model.

        Returns:
            dimension: Vector dimension reported by the model.
        """
        return self._load_model().get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encodes texts into a 2D array of dense vectors.

        Args:
            texts: Raw texts to encode.

        Returns:
            vectors: One vector per input text, shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        model = self._load_model()
        logger.debug("encoding %d texts with %s", len(texts), self.config.model_name)
        return model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )

    def _load_model(self) -> SentenceTransformer:
        """Loads the model on first use.

        Returns:
            model: Loaded sentence-transformers model.
        """
        if self._model is None:
            logger.info("loading embedding model %s (device=%s)", self.config.model_name, self.config.device)
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model


    def encode_query(self, query: str) -> np.ndarray:
        """Encodes one query text into a single vector.

        Args:
            query: Raw query text.

        Returns:
            vector: Dense vector for the query.
        """
        return self.encode([query])[0]

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        """Encodes document texts into dense vectors.

        Args:
            documents: Raw document texts.

        Returns:
            vectors: One vector per document.
        """
        return self.encode(documents)
