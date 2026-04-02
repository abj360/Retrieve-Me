#!/usr/bin/env python3
"""
test_ingestion_smoke.py --- smoke tests for the ingestion pipeline

Contains:
    MockEmbedder: stand-in embedder that never inspects its input
    smoke_documents(): two small documents for smoke runs
    test_ingest_smoke_writes_chunks(): asserts ingest writes without errors
"""

import numpy as np
import pytest

from src.ingest.bm25_index import BM25Index
from src.ingest.loader import CorpusIngestor, Document


class MockEmbedder:
    """Stands in for the embedder during smoke runs; never inspects input."""

    def __init__(self) -> None:
        """Creates the mock with a call counter."""
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns a constant vector per text regardless of content.

        Args:
            texts: Ignored input texts.

        Returns:
            vectors: Constant vectors, one per text.
        """
        self.calls += 1
        return np.full((len(texts), 16), 0.5, dtype=np.float32)


@pytest.fixture
def smoke_documents() -> list[Document]:
    """Returns two small documents for smoke runs."""
    return [
        Document(doc_id="smoke-legal", title="smoke-legal", text="Section 1.1 The tenant shall pay rent."),
        Document(doc_id="smoke-tech", title="smoke-tech", text="The API returns problem details as JSON."),
    ]


def test_ingest_smoke_writes_chunks(smoke_documents, whole_doc_chunker, fake_dense_index) -> None:
    """Asserts an ingest run writes chunks and calls the embedder."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingested = ingestor.ingest(smoke_documents)
    assert ingested > 0
    assert embedder.calls > 0
    assert fake_dense_index.count() > 0
