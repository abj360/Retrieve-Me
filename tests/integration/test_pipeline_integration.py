#!/usr/bin/env python3
"""
test_pipeline_integration.py --- integration tests for the retrieval pipeline

Contains:
    indexed_stores(): ingests the sample corpus into fake stores
    test_exact_clause_doc_wins_fused_ranking(): asserts hybrid beats dense-only
    test_sparse_leg_finds_exact_terms(): asserts bm25 hits exact clause text
    test_dense_leg_returns_scored_hits(): asserts dense search returns scores
"""

import pytest

from src.ingest.bm25_index import BM25Index
from src.retrieval.fusion import FusionConfig, ResultFuser


@pytest.fixture
def indexed_stores(sample_documents, whole_doc_chunker, stub_embedder, fake_dense_index):
    """Ingests the sample documents into the fake dense index and a real bm25.

    Args:
        sample_documents: Small legal/tech corpus fixture.
        whole_doc_chunker: Test chunker keeping documents whole.
        stub_embedder: Deterministic hashing embedder fixture.
        fake_dense_index: In-memory dense index double.

    Returns:
        stores: (bm25_index, dense_index) with the corpus indexed.
    """
    chunks = [
        chunk
        for doc_id, text in sample_documents
        for chunk in whole_doc_chunker.split(text, doc_id)
    ]
    vectors = stub_embedder.encode([chunk.text for chunk in chunks])
    fake_dense_index.ensure_collection(recreate=True)
    fake_dense_index.upsert(
        [chunk.chunk_id for chunk in chunks],
        vectors,
        [{"doc_id": chunk.doc_id, "text": chunk.text} for chunk in chunks],
    )
    bm25 = BM25Index()
    bm25.build(chunks)
    return bm25, fake_dense_index


def test_exact_clause_doc_wins_fused_ranking(indexed_stores, stub_embedder) -> None:
    """Asserts the fused ranking puts the exact-clause document first."""
    bm25, dense = indexed_stores
    query = "Section 3.1 indemnify vendor"
    sparse_hits = bm25.search(query, top_k=4)
    from src.retrieval.fusion import RankedResult

    sparse = [
        RankedResult(
            chunk_id=hit.chunk_id, doc_id=hit.doc_id, text=hit.text, score=hit.score, source="sparse"
        )
        for hit in sparse_hits
    ]
    query_vector = stub_embedder.encode([query])[0]
    dense_hits = dense.search(query_vector, top_k=4)
    dense_results = [
        RankedResult(
            chunk_id=hit.chunk_id,
            doc_id=hit.payload["doc_id"],
            text=hit.payload["text"],
            score=hit.score,
            source="dense",
        )
        for hit in dense_hits
    ]
    fused = ResultFuser(FusionConfig()).fuse(sparse, dense_results)
    assert fused[0].doc_id == "license-agreement"


def test_sparse_leg_finds_exact_terms(indexed_stores) -> None:
    """Asserts bm25 finds the document containing the exact clause text."""
    bm25, _dense = indexed_stores
    hits = bm25.search("RFC 7807 problem details", top_k=2)
    assert hits[0].doc_id == "rfc-7807"


def test_dense_leg_returns_scored_hits(indexed_stores, stub_embedder) -> None:
    """Asserts the dense leg returns cosine scores in [0, 1]."""
    _bm25, dense = indexed_stores
    query_vector = stub_embedder.encode(["connection pooling"])[0]
    hits = dense.search(query_vector, top_k=3)
    assert hits
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


def test_fusion_is_deterministic(indexed_stores, stub_embedder) -> None:
    """Asserts repeated fusion of the same lists yields the same order."""
    bm25, dense = indexed_stores
    from src.retrieval.fusion import FusionConfig, RankedResult, ResultFuser

    query = "indemnify"
    sparse = [
        RankedResult(hit.chunk_id, hit.doc_id, hit.text, hit.score, "sparse")
        for hit in bm25.search(query, top_k=3)
    ]
    dense_hits = dense.search(stub_embedder.encode([query])[0], top_k=3)
    dense_results = [
        RankedResult(hit.chunk_id, hit.payload["doc_id"], hit.payload["text"], hit.score, "dense")
        for hit in dense_hits
    ]
    fuser = ResultFuser(FusionConfig())
    first = [r.chunk_id for r in fuser.fuse(sparse, dense_results)]
    second = [r.chunk_id for r in fuser.fuse(sparse, dense_results)]
    assert first == second


def test_stub_embedder_is_deterministic(stub_embedder) -> None:
    """Asserts the stub embedder encodes the same text identically twice."""
    first = stub_embedder.encode(["determinism check"])[0]
    second = stub_embedder.encode(["determinism check"])[0]
    assert list(first) == list(second)


def test_fake_dense_cosine_orthogonal_is_zero(fake_dense_index) -> None:
    """Asserts orthogonal vectors score zero in the fake dense index."""
    fake_dense_index.upsert(["a"], [[1.0, 0.0]], [{"doc_id": "a"}])
    hits = fake_dense_index.search([0.0, 1.0], top_k=1)
    assert hits[0].score == 0.0


def test_fake_dense_upsert_reports_written_count(fake_dense_index) -> None:
    """Asserts upsert returns the number of points written."""
    written = fake_dense_index.upsert(["a", "b"], [[0.1, 0.2], [0.3, 0.4]], [{"d": 1}, {"d": 2}])
    assert written == 2
    assert fake_dense_index.count() == 2


def test_bm25_ranks_exact_clause_first(indexed_stores) -> None:
    """Asserts bm25 puts the exact-clause chunk above looser matches."""
    bm25, _dense = indexed_stores
    hits = bm25.search("Section 3.1", top_k=2)
    assert hits[0].doc_id == "license-agreement"


def test_fused_results_carry_source(indexed_stores, stub_embedder) -> None:
    """Asserts fused results are labelled with the fused source."""
    bm25, dense = indexed_stores
    from src.retrieval.fusion import FusionConfig, RankedResult, ResultFuser

    query = "batch upserts"
    sparse = [
        RankedResult(hit.chunk_id, hit.doc_id, hit.text, hit.score, "sparse")
        for hit in bm25.search(query, top_k=3)
    ]
    dense_results = [
        RankedResult(hit.chunk_id, hit.payload["doc_id"], hit.payload["text"], hit.score, "dense")
        for hit in dense.search(stub_embedder.encode([query])[0], top_k=3)
    ]
    fused = ResultFuser(FusionConfig()).fuse(sparse, dense_results)
    assert all(result.source == "fused" for result in fused)


def test_reranker_reorders_by_overlap(indexed_stores, stub_reranker) -> None:
    """Asserts the stub reranker lifts the highest-overlap candidate."""
    bm25, _dense = indexed_stores
    from src.retrieval.fusion import RankedResult

    query = "problem details HTTP APIs"
    sparse = [
        RankedResult(
            chunk_id=hit.chunk_id, doc_id=hit.doc_id, text=hit.text, score=hit.score, source="sparse"
        )
        for hit in bm25.search(query, top_k=3)
    ]
    reranked = stub_reranker.rerank(query, sparse)
    assert reranked[0].doc_id == "rfc-7807"


def test_token_chunker_respects_small_budget(sample_documents, token_chunker) -> None:
    """Asserts the small test budget yields multiple chunks per document."""
    chunks = token_chunker.split(sample_documents[0][1], "license-agreement")
    assert len(chunks) >= 2
    assert all(chunk.token_count <= 48 for chunk in chunks)


def test_whole_doc_chunker_yields_one_chunk(sample_documents, whole_doc_chunker) -> None:
    """Asserts the whole-doc chunker emits exactly one chunk per document."""
    chunks = whole_doc_chunker.split(sample_documents[0][1], "license-agreement")
    assert len(chunks) == 1


def test_semantic_chunker_preserves_clause_text(sample_documents, token_chunker) -> None:
    """Asserts clause text survives chunking intact for exact-match lookup."""
    chunks = token_chunker.split(sample_documents[0][1], "license-agreement")
    assert any("Section 3.1" in chunk.text for chunk in chunks)
    assert all(chunk.token_count <= 48 for chunk in chunks)


def test_semantic_chunks_cover_full_document(sample_documents, token_chunker) -> None:
    """Asserts semantic chunks cover the document without dropping content."""
    _doc_id, text = sample_documents[0]
    chunks = token_chunker.split(text, "license-agreement")
    for phrase in ("Section 3.1", "Section 3.2"):
        assert any(phrase in chunk.text for chunk in chunks)


def test_semantic_chunk_ids_stable(sample_documents, token_chunker) -> None:
    """Asserts repeated splitting yields identical chunk ids."""
    _doc_id, text = sample_documents[0]
    first = [chunk.chunk_id for chunk in token_chunker.split(text, "license-agreement")]
    second = [chunk.chunk_id for chunk in token_chunker.split(text, "license-agreement")]
    assert first == second


def test_semantic_pack_order_preserved(sample_documents, token_chunker) -> None:
    """Asserts chunks come out in document order."""
    _doc_id, text = sample_documents[0]
    chunks = token_chunker.split(text, "license-agreement")
    assert [chunk.index for chunk in chunks] == sorted(chunk.index for chunk in chunks)


def test_dense_filters_exclude_nonmatching(indexed_stores, stub_embedder) -> None:
    """Asserts payload filters narrow dense hits to matching documents."""
    _bm25, dense = indexed_stores
    query_vector = stub_embedder.encode(["indemnify"])[0]
    hits = dense.search(query_vector, top_k=4, filters={"doc_id": "rfc-7807"})
    assert all(hit.payload["doc_id"] == "rfc-7807" for hit in hits)
