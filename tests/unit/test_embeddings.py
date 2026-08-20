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

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.retrieval.embeddings import (
    DeterministicEmbedder,
    EmbeddingConfig,
    SentenceTransformerEmbedder,
    batched,
)


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
    with pytest.raises(FrozenInstanceError):
        config.batch_size = 8


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


def test_encode_query_returns_single_vector() -> None:
    """Asserts encode_query returns a 1D vector."""
    embedder, _fake = make_embedder()
    vector = embedder.encode_query("solo")
    assert vector.shape == (8,)


def test_encode_documents_matches_encode() -> None:
    """Asserts encode_documents matches encode output."""
    embedder, _fake = make_embedder()
    docs = ["one", "two"]
    assert embedder.encode_documents(docs).shape == embedder.encode(docs).shape


def test_deterministic_embedder_same_text_same_vector() -> None:
    """Asserts the deterministic embedder is stable per text."""
    embedder = DeterministicEmbedder(dimension=16)
    first = embedder.encode(["stable text"])[0]
    second = embedder.encode(["stable text"])[0]
    assert list(first) == list(second)


def test_deterministic_embedder_different_texts_differ() -> None:
    """Asserts different texts get different deterministic vectors."""
    embedder = DeterministicEmbedder(dimension=16)
    first, second = embedder.encode(["alpha", "beta"])
    assert list(first) != list(second)


def test_deterministic_embedder_unit_norm() -> None:
    """Asserts deterministic vectors are L2-normalized by default."""
    vector = DeterministicEmbedder(dimension=16).encode(["norm check"])[0]
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_deterministic_embedder_query_matches_encode() -> None:
    """Asserts encode_query equals encode of the singleton list."""
    embedder = DeterministicEmbedder(dimension=16)
    assert list(embedder.encode_query("q")) == list(embedder.encode(["q"])[0])


def test_encode_uses_batched_iteration() -> None:
    """Asserts encode iterates in config-sized batches."""
    embedder, fake = make_embedder()
    embedder.encode([f"text-{index}" for index in range(9)])
    assert fake.encode_calls and sum(fake.encode_calls) == 9


def test_encode_concatenates_batch_results() -> None:
    """Asserts batch results concatenate into one array."""
    embedder, _fake = make_embedder()
    vectors = embedder.encode([f"text-{index}" for index in range(9)])
    assert vectors.shape == (9, 8)


def test_warmup_loads_model_once() -> None:
    """Asserts warmup is idempotent."""
    embedder = DeterministicEmbedder()
    embedder.encode(["warm"])
    embedder.encode(["warm"])
    first = embedder.encode(["warm"])[0]
    second = embedder.encode(["warm"])[0]
    assert list(first) == list(second)


def test_unnormalized_option() -> None:
    """Asserts normalization can be disabled."""
    vector = DeterministicEmbedder(dimension=16, normalize=False).encode(["raw"])[0]
    assert np.linalg.norm(vector) != pytest.approx(1.0)


def test_deterministic_dimension_property() -> None:
    """Asserts the deterministic embedder reports its dimension."""
    assert DeterministicEmbedder(dimension=32).dimension == 32


def test_batched_size_one() -> None:
    """Asserts batch_size of one yields singleton batches."""
    assert [list(batch) for batch in batched(["a", "b"], 1)] == [["a"], ["b"]]


def test_config_custom_values() -> None:
    """Asserts EmbeddingConfig accepts overrides."""
    config = EmbeddingConfig(model_name="other-model", batch_size=8, normalize=False)
    assert config.model_name == "other-model"
    assert config.batch_size == 8


def test_deterministic_embedder_documents_entrypoint() -> None:
    """Asserts encode_documents works on the deterministic embedder."""
    vectors = DeterministicEmbedder(dimension=8).encode_documents(["a", "b"])
    assert vectors.shape == (2, 8)


def test_batched_empty_input() -> None:
    """Asserts batched yields nothing for empty input."""
    assert list(batched([], 4)) == []
