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


def test_ingest_crlf_document(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts CRLF line endings ingest without issues."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="crlf", title="crlf", text="Clause 3.1 First line.\r\nSecond line.")]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert fake_dense_index.count() == 1


def test_ingest_markdown_document(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts markdown-formatted text ingests without issues."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="md", title="md", text="# Heading\n\n- item one\n- item two")]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert fake_dense_index.count() == 1


def test_ingest_bullet_list_document(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts bullet-list text ingests without issues."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    text = "\n".join(f"- obligation {index}" for index in range(20))
    documents = [Document(doc_id="bullets", title="bullets", text=text)]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert fake_dense_index.count() == 1


def test_ingest_numbered_clauses_document(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts numbered-clause text ingests without issues."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    text = " ".join(f"Clause {index}.1 obligation {index}." for index in range(10))
    documents = [Document(doc_id="clauses", title="clauses", text=text)]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert fake_dense_index.count() == 1


def test_ingest_long_document(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts a long document ingests without issues."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    text = "The quick brown fox jumps over the lazy dog. " * 400
    documents = [Document(doc_id="long", title="long", text=text)]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert fake_dense_index.count() == 1


def test_ingest_embedder_called_once_per_small_corpus(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts a small corpus triggers one embed call per batch."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id=f"d-{index}", title="t", text=f"text {index}") for index in range(5)]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert embedder.calls == 1


def test_ingest_chunk_ids_unique(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts ingested chunks get distinct ids."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id=f"doc-{index}", title="t", text=f"unique text {index}") for index in range(4)]
    CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25).ingest(documents)
    assert fake_dense_index.count() == 4


def test_ingest_single_sentence_document(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts a one-sentence document ingests as a single chunk."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="tiny", title="tiny", text="Clause 1.1 Short.")]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingested = ingestor.ingest(documents)
    assert ingested == 1
    assert fake_dense_index.count() == 1


def test_ingest_handles_unicode(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts unicode text ingests without errors."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="uni", title="uni", text="Clause 2.1 — café “smart” ünicode.")]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingested = ingestor.ingest(documents)
    assert fake_dense_index.count() == 1


def test_ingest_unicode_quotes_and_emoji(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts emoji and smart quotes ingest without issues."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="emoji", title="emoji", text="Clause 4.1 ship it 🚀 “done”")]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert fake_dense_index.count() == 1


def test_ingest_long_token_stream(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts a long single-token stream does not hang ingest."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="token", title="token", text="x" * 5000)]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert fake_dense_index.count() == 1
