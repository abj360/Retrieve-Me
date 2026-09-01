#!/usr/bin/env python3
"""
test_sparse_index_persistence.py --- covers the sparse index surviving the process

Contains:
    make_documents(): small corpus for round-trip tests
    test_ingest_persists_the_sparse_index(): asserts ingest writes the index file
    test_ingest_skips_persistence_without_a_path(): asserts the path stays optional
    test_service_loads_the_persisted_index(): asserts a fresh load answers searches
    test_missing_index_degrades_instead_of_raising(): asserts no file still serves
    test_sparse_index_is_a_readiness_check(): asserts readyz reports a missing index
"""

from pathlib import Path

from src.api.dependencies import Settings, load_bm25_index
from src.api.routers.health import check_sparse_index
from src.ingest.bm25_index import BM25Index
from src.ingest.loader import CorpusIngestor, Document


def make_documents() -> list[Document]:
    """Returns a small corpus with a findable exact clause.

    Returns:
        documents: Two documents, one carrying "Section 3.1".
    """
    return [
        Document(
            doc_id="license",
            title="license",
            text="Section 3.1 The licensee shall indemnify the vendor against claims.",
        ),
        Document(
            doc_id="release",
            title="release",
            text="Release 2.4 adds batch upserts to the vector store client.",
        ),
    ]


def test_ingest_persists_the_sparse_index(
    tmp_path, whole_doc_chunker, fake_dense_index, real_embedder
) -> None:
    """Asserts ingest writes the index where the service expects to find it."""
    index_path = tmp_path / "nested" / "bm25_index.pkl"
    CorpusIngestor(
        whole_doc_chunker,
        real_embedder,
        fake_dense_index,
        BM25Index(),
        bm25_index_path=index_path,
    ).ingest(make_documents())
    assert index_path.exists()


def test_ingest_skips_persistence_without_a_path(
    whole_doc_chunker, fake_dense_index, real_embedder
) -> None:
    """Asserts omitting the path leaves ingest working and writes nothing."""
    stats = CorpusIngestor(whole_doc_chunker, real_embedder, fake_dense_index, BM25Index()).ingest(
        make_documents()
    )
    assert stats.chunks == 2


def test_service_loads_the_persisted_index(
    tmp_path, whole_doc_chunker, fake_dense_index, real_embedder
) -> None:
    """Asserts an index built in one process answers searches after reloading."""
    index_path = tmp_path / "bm25_index.pkl"
    CorpusIngestor(
        whole_doc_chunker,
        real_embedder,
        fake_dense_index,
        BM25Index(),
        bm25_index_path=index_path,
    ).ingest(make_documents())

    reloaded = load_bm25_index(index_path)
    hits = reloaded.search("Section 3.1 indemnify", top_k=2)
    assert hits and hits[0].doc_id == "license"


def test_missing_index_degrades_instead_of_raising(tmp_path) -> None:
    """Asserts a missing index yields an empty searchable index, not a crash."""
    index = load_bm25_index(tmp_path / "absent.pkl")
    assert index.search("anything", top_k=5) == []


def test_sparse_index_is_a_readiness_check(tmp_path) -> None:
    """Asserts readiness reports the index missing before a query hits it."""
    settings = Settings(bm25_index_path=tmp_path / "absent.pkl")
    assert check_sparse_index(settings) == "missing"

    present = tmp_path / "present.pkl"
    present.write_bytes(b"")
    assert check_sparse_index(Settings(bm25_index_path=present)) == "ok"


def test_default_index_path_is_under_data(tmp_path: Path) -> None:
    """Asserts the default location matches what the Dockerfile ships."""
    assert Settings().bm25_index_path == Path("data/bm25_index.pkl")
