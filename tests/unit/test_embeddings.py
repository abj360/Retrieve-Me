#!/usr/bin/env python3
"""
test_embeddings.py --- unit tests for the dense embedding pipeline

Contains:
    FakeModel: sentence-transformers double recording encode calls
    make_embedder(): builds an embedder wired to the fake model
    test_batched_even_split(): asserts even batches
    test_batched_uneven_split(): asserts the short tail batch
    test_config_defaults(): asserts EmbeddingConfig defaults
"""

import numpy as np
import pytest

from src.retrieval.embeddings import EmbeddingConfig, SentenceTransformerEmbedder, batched


class FakeModel:
    """Sentence-transformers double recording encode calls."""

    def __init__(self, dimension: int = 8) -> None:
        """Creates the fake with a fixed output dimension.

        Args:
            dimension: Vector dimension reported and produced.
        """
        self._dimension = dimension
        self.encode_calls: list[int] = []

    def get_sentence_embedding_dimension(self) -> int:
        """Returns the fake dimension.

        Returns:
            dimension: Vector dimension.
        """
        return self._dimension

    def encode(self, texts, batch_size: int, normalize_embeddings: bool, convert_to_numpy: bool):
        """Records the call and returns deterministic vectors.

        Args:
            texts: Texts to encode.
            batch_size: Batch size passed through.
            normalize_embeddings: Whether to normalize.
            convert_to_numpy: Whether to return numpy.

        Returns:
            vectors: Ones matrix of the right shape.
        """
        self.encode_calls.append(len(texts))
        return np.ones((len(texts), self._dimension), dtype=np.float32)


def make_embedder(dimension: int = 8) -> tuple[SentenceTransformerEmbedder, FakeModel]:
    """Builds an embedder wired to the fake model.

    Args:
        dimension: Fake output dimension.

    Returns:
        pair: (embedder, fake model) for assertions.
    """
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(batch_size=4))
    fake = FakeModel(dimension=dimension)
    embedder._model = fake
    return embedder, fake


def test_batched_even_split() -> None:
    """Asserts texts split into even full batches."""
    assert [list(batch) for batch in batched(["a", "b", "c", "d"], 2)] == [["a", "b"], ["c", "d"]]


def test_batched_uneven_split() -> None:
    """Asserts the tail batch carries the remainder."""
    assert [list(batch) for batch in batched(["a", "b", "c"], 2)] == [["a", "b"], ["c"]]


def test_config_defaults() -> None:
    """Asserts EmbeddingConfig defaults."""
    config = EmbeddingConfig()
    assert config.batch_size == 32
    assert config.normalize is True
    assert config.device is None


def test_config_is_immutable() -> None:
    """Asserts EmbeddingConfig is frozen."""
    config = EmbeddingConfig()
    with pytest.raises(Exception):
        config.batch_size = 8  # noqa: frozen dataclass guard


def test_encode_returns_expected_shape() -> None:
    """Asserts encode returns one vector per input text."""
    embedder, _fake = make_embedder()
    vectors = embedder.encode(["a", "b", "c"])
    assert vectors.shape == (3, 8)


def test_encode_empty_returns_zero_rows() -> None:
    """Asserts empty input yields a (0, dimension) array."""
    embedder, _fake = make_embedder()
    vectors = embedder.encode([])
    assert vectors.shape == (0, 8)


def test_dimension_property() -> None:
    """Asserts the dimension property reads the model dimension."""
    embedder, _fake = make_embedder(dimension=16)
    assert embedder.dimension == 16


def test_encode_delegates_to_model() -> None:
    """Asserts encode forwards texts to the underlying model."""
    embedder, fake = make_embedder()
    embedder.encode(["x", "y"])
    assert fake.encode_calls
