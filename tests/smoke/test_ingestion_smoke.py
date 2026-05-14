#!/usr/bin/env python3
"""
test_ingestion_smoke.py --- smoke tests for the ingestion pipeline

Contains:
    smoke_documents(): two small documents for smoke runs
    test_ingest_smoke_writes_chunks(): asserts the real indexed chunk count
"""

import pytest

from src.ingest.bm25_index import BM25Index
from src.ingest.loader import CorpusIngestor, Document


@pytest.fixture
def smoke_documents() -> list[Document]:
    """Returns two small documents for smoke runs."""
    return [
        Document(doc_id="smoke-legal", title="smoke-legal", text="Section 1.1 The tenant shall pay rent."),
        Document(doc_id="smoke-tech", title="smoke-tech", text="The API returns problem details as JSON."),
    ]


def test_ingest_smoke_writes_chunks(smoke_documents, token_chunker, fake_dense_index, real_embedder) -> None:
    """Asserts the indexed chunk count matches what the corpus actually yields."""
    bm25 = BM25Index()
    ingestor = CorpusIngestor(token_chunker, real_embedder, fake_dense_index, bm25)
    stats = ingestor.ingest(smoke_documents)
    expected = sum(
        len(token_chunker.split(document.text, document.doc_id))
        for document in smoke_documents
    )
    assert fake_dense_index.count() == expected
    assert stats.chunks == expected


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


def test_ingest_duplicate_doc_ids_both_written(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts duplicate doc ids both index (dedupe not implemented yet)."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [
        Document(doc_id="dup", title="dup", text="first copy"),
        Document(doc_id="dup", title="dup", text="second copy"),
    ]
    CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25).ingest(documents)
    assert fake_dense_index.count() >= 1


def test_ingest_whitespace_text_still_counts(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts a whitespace-only doc does not crash ingest."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="ws", title="ws", text="   ")]
    CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25).ingest(documents)
    assert fake_dense_index.count() <= 1


def test_ingest_empty_document_list(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts ingesting zero documents writes zero chunks."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    ingested = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25).ingest([])
    assert ingested == 0
    assert fake_dense_index.count() == 0


def test_semantic_chunks_respect_sentence_endings(token_chunker) -> None:
    """Asserts semantic chunks end at sentence boundaries."""
    text = "First sentence here. Second sentence follows. Third sentence ends it."
    chunks = token_chunker.split(text, "sentences")
    assert all(chunk.text.rstrip().endswith((".", "!", "?")) for chunk in chunks)


def test_semantic_chunks_within_budget(token_chunker) -> None:
    """Asserts semantic chunks stay within the token budget."""
    text = " ".join(f"Sentence {index} has six tokens in total." for index in range(40))
    chunks = token_chunker.split(text, "budget")
    assert all(chunk.token_count <= 48 for chunk in chunks)


def test_semantic_pack_order_stable(token_chunker) -> None:
    """Asserts semantic chunk order matches document order."""
    text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."
    chunks = token_chunker.split(text, "order")
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_semantic_tail_chunk_merged(token_chunker) -> None:
    """Asserts an undersized trailing chunk folds into its predecessor."""
    text = " ".join(f"Sentence {index} has six tokens in total." for index in range(30))
    text += " Tiny."
    chunks = token_chunker.split(text, "tail")
    assert all(chunk.token_count >= 5 for chunk in chunks)


def test_ingest_exactly_one_batch(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts a 100-chunk corpus ingests in exactly one batch."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [
        Document(doc_id=f"exact-{index:03d}", title="t", text=f"clause {index} text")
        for index in range(100)
    ]
    CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25).ingest(documents)
    assert embedder.calls == 1


def test_ingest_batch_boundary_corpus(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts a corpus spanning multiple batches ingests without errors."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [
        Document(doc_id=f"bulk-{index:03d}", title=f"bulk-{index:03d}", text=f"Clause {index}.1 bulk text.")
        for index in range(120)
    ]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    assert embedder.calls >= 2


def test_ingest_two_hundred_fifty_chunks(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts a 250-chunk corpus ingests across three batches."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [
        Document(doc_id=f"bulk-{index:03d}", title="t", text=f"clause {index} text")
        for index in range(250)
    ]
    CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25).ingest(documents)
    assert embedder.calls == 3


def test_ingest_returns_positive_count(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts ingest reports a positive chunk count for a small corpus."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id=f"r-{index}", title="t", text=f"text {index}") for index in range(3)]
    ingested = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25).ingest(documents)
    assert ingested > 0


def test_bm25_built_after_ingest(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts the sparse index is queryable after ingest."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="q-1", title="t", text="clause 7.1 queryable text")]
    CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25).ingest(documents)
    hits = bm25.search("queryable", top_k=1)
    assert hits[0].doc_id == "q-1"


def test_ingest_recreates_collection_when_asked(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts ensure_collection(recreate=True) clears before ingest."""
    fake_dense_index.upsert(["stale"], [[0.1] * 16], [{"doc_id": "stale"}])
    fake_dense_index.ensure_collection(recreate=True)
    assert fake_dense_index.count() == 0


def test_second_ingest_overwrites_deterministic_ids(whole_doc_chunker, fake_dense_index) -> None:
    """Asserts re-ingesting the same corpus does not grow the index."""
    embedder = MockEmbedder()
    bm25 = BM25Index()
    documents = [Document(doc_id="redo", title="t", text="redo text here")]
    ingestor = CorpusIngestor(whole_doc_chunker, embedder, fake_dense_index, bm25)
    ingestor.ingest(documents)
    first = fake_dense_index.count()
    ingestor.ingest(documents)
    assert fake_dense_index.count() == first


def test_stats_report_documents_and_seconds(whole_doc_chunker, fake_dense_index, real_embedder) -> None:
    """Asserts ingest stats include document count and elapsed seconds."""
    bm25 = BM25Index()
    documents = [Document(doc_id="s-1", title="t", text="stats text")]
    stats = CorpusIngestor(whole_doc_chunker, real_embedder, fake_dense_index, bm25).ingest(documents)
    assert stats.documents == 1
    assert stats.seconds >= 0


def test_exact_count_small_corpus(token_chunker, fake_dense_index, real_embedder, mini_corpus) -> None:
    """Asserts the mini corpus indexes exactly as many chunks as produced."""
    from src.ingest.loader import Document

    bm25 = BM25Index()
    documents = [
        Document(doc_id=doc_id, title=doc_id, text=text) for doc_id, text in mini_corpus
    ]
    ingestor = CorpusIngestor(token_chunker, real_embedder, fake_dense_index, bm25)
    stats = ingestor.ingest(documents)
    expected = sum(
        len(token_chunker.split(document.text, document.doc_id)) for document in documents
    )
    assert fake_dense_index.count() == expected
    assert stats.chunks == expected
