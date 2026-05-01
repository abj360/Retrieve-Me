#!/usr/bin/env python3
"""
embeddings.py --- dense text embedding pipeline backed by sentence-transformers

Contains:
    EmbeddingConfig: immutable settings for the embedding model
    batched(): splits a sequence into fixed-size batches
    SentenceTransformerEmbedder: lazily loads the model and encodes texts
    SentenceTransformerEmbedder.encode_query(): encodes one query into a vector
    SentenceTransformerEmbedder.encode_documents(): encodes documents into vectors
    DeterministicEmbedder: hashing-based embedder for offline dev and test suites
"""

import hashlib
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
        device: Torch device identifier; None lets sentence-transformers choose.
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
        logger.debug("encoding %d texts (batch_size=%d)", len(texts), self.config.batch_size)
        vectors = []
        logger.debug("encoding %d texts in batches of %d", len(texts), self.config.batch_size)
        for batch in batched(texts, self.config.batch_size):
            vectors.append(
                model.encode(
                    list(batch),
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize,
                    convert_to_numpy=True,
                )
            )
        return np.concatenate(vectors)

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


class DeterministicEmbedder:
    """Produces deterministic pseudo-embeddings from text hashes for offline dev/tests.

    Attributes:
        dimension: Dimensionality of the produced vectors.
    """

    def __init__(self, dimension: int = 384, normalize: bool = True) -> None:
        """Stores the vector dimension and normalization flag.

        Args:
            dimension: Dimensionality of the produced vectors.
            normalize: Whether to L2-normalize output vectors.
        """
        self._dimension = dimension
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        """Returns the vector dimensionality.

        Returns:
            dimension: Vector dimension configured at construction.
        """
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encodes texts into deterministic hash-derived vectors.

        Args:
            texts: Raw texts to encode.

        Returns:
            vectors: One deterministic vector per input text.
        """
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeats = (self._dimension + len(digest) - 1) // len(digest)
            values = np.frombuffer(digest * repeats, dtype=np.uint8)[: self._dimension]
            vectors[row] = (values.astype(np.float32) - 128.0) / 128.0
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes one query into a deterministic vector.

        Args:
            query: Raw query text.

        Returns:
            vector: Deterministic vector for the query.
        """
        return self.encode([query])[0]

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        """Encodes documents into deterministic vectors.

        Args:
            documents: Raw document texts.

        Returns:
            vectors: One deterministic vector per document.
        """
        return self.encode(documents)
