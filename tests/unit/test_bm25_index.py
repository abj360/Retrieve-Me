#!/usr/bin/env python3
"""
test_bm25_index.py --- unit tests for the BM25 sparse index

Contains:
    make_chunk(): builds a minimal chunk-like object for indexing
    test_tokenize_lowercases_and_keeps_hyphens(): asserts tokenizer behavior
    test_tokenize_lowercases_and_keeps_hyphens(): asserts tokenizer behavior
    test_search_ranks_relevant_chunk_first(): asserts bm25 ranks exact terms well
    test_search_before_build_raises(): asserts the index must be built first
    test_save_load_roundtrip(): asserts pickled indexes keep ranking
"""

from types import SimpleNamespace

import pytest

from src.ingest.bm25_index import BM25Index, token_count, tokenize


def make_chunk(chunk_id: str, text: str) -> SimpleNamespace:
    """Builds a minimal chunk-like object for indexing.

    Args:
        chunk_id: Identifier for the chunk.
        text: Chunk text to index.

    Returns:
        chunk: Object exposing chunk_id, doc_id, and text.
    """
    return SimpleNamespace(chunk_id=chunk_id, doc_id=chunk_id.split("-", maxsplit=1)[0], text=text)


def test_tokenize_lowercases_and_keeps_hyphens() -> None:
    """Asserts the tokenizer lowercases and emits both forms of a hyphenated term."""
    assert tokenize("Multi-Agent SYSTEMS, Inc.") == [
        "multi-agent",
        "multi",
        "agent",
        "systems",
        "inc",
    ]


def test_search_ranks_relevant_chunk_first() -> None:
    """Asserts bm25 ranks the exact-term chunk above unrelated text."""
    index = BM25Index()
    index.build(
        [
            make_chunk("license-chunk-0", "Section 3.1 the licensee shall indemnify the vendor"),
            make_chunk("rfc-chunk-0", "RFC 7807 defines problem details for HTTP APIs"),
        ]
    )
    hits = index.search("indemnify", top_k=2)
    assert hits[0].chunk_id == "license-chunk-0"


def test_search_before_build_raises() -> None:
    """Asserts searching before build() fails loudly."""
    with pytest.raises(RuntimeError):
        BM25Index().search("anything", top_k=3)


def test_tokenize_empty_string() -> None:
    """Asserts tokenizing empty text yields no terms."""
    assert tokenize("") == []


def test_save_load_roundtrip(tmp_path) -> None:
    """Asserts a pickled index keeps ranking after a round trip."""
    index = BM25Index()
    index.build([make_chunk("a-0", "alpha beta gamma"), make_chunk("b-0", "delta epsilon")])
    target = tmp_path / "bm25.pkl"
    index.save(target)
    restored = BM25Index()
    restored.load(target)
    assert restored.search("alpha", top_k=1)[0].chunk_id == "a-0"


def test_empty_index_returns_no_hits() -> None:
    """Asserts searching an empty built index returns []."""
    index = BM25Index()
    index.build([])
    assert index.search("anything", top_k=3) == []


def test_blank_query_returns_no_hits() -> None:
    """Asserts a whitespace-only query short-circuits to []."""
    index = BM25Index()
    index.build([make_chunk("a-0", "alpha beta")])
    assert index.search("   ", top_k=2) == []


def test_tokenize_strips_punctuation() -> None:
    """Asserts punctuation is dropped from terms."""
    assert tokenize("clause, section; article.") == ["clause", "section", "article"]


def test_stats_reports_chunk_count() -> None:
    """Asserts stats() counts chunks and average tokens."""
    index = BM25Index()
    index.build([make_chunk("a-0", "one two three"), make_chunk("b-0", "four five")])
    stats = index.stats()
    assert stats["chunks"] == 2
    assert stats["avg_tokens"] == 2.5


def test_token_count_matches_tokenize_length() -> None:
    """Asserts token_count agrees with len(tokenize())."""
    assert token_count("one two three") == len(tokenize("one two three"))


def hyphenation_corpus() -> list:
    """Returns a corpus large enough for BM25 idf to be meaningful.

    A term present in exactly half a corpus scores idf zero under Okapi, so
    these tests need more than a pair of documents to say anything.

    Returns:
        chunks: Six chunks, one hyphenated and one split-form target.
    """
    return [
        make_chunk("hyphenated-0", "The cross-encoder reranks the fused candidate set."),
        make_chunk("split-0", "The multi agent planner bounds its own loop."),
        make_chunk("filler-0", "Connection pooling is bounded to prevent exhaustion."),
        make_chunk("filler-1", "Section 3.1 covers indemnification obligations."),
        make_chunk("filler-2", "RFC 7807 defines problem details for HTTP APIs."),
        make_chunk("filler-3", "Release notes list the batch upsert change."),
    ]


def test_hyphenated_term_indexed_whole_and_split() -> None:
    """Asserts a hyphenated term yields the joined form and both parts."""
    assert tokenize("cross-encoder") == ["cross-encoder", "cross", "encoder"]


def test_unhyphenated_query_finds_hyphenated_document() -> None:
    """Asserts 'cross encoder' matches a document written 'cross-encoder'."""
    index = BM25Index()
    index.build(hyphenation_corpus())
    hits = index.search("cross encoder", top_k=3)
    assert hits[0].chunk_id == "hyphenated-0"
    assert hits[0].score > 0


def test_hyphenated_query_finds_unhyphenated_document() -> None:
    """Asserts 'multi-agent' matches a document written 'multi agent'."""
    index = BM25Index()
    index.build(hyphenation_corpus())
    hits = index.search("multi-agent", top_k=3)
    assert hits[0].chunk_id == "split-0"
    assert hits[0].score > 0


def test_exact_hyphenated_match_still_outranks_a_partial_one() -> None:
    """Asserts splitting does not let a one-word partial beat the exact term."""
    index = BM25Index()
    index.build([*hyphenation_corpus(), make_chunk("partial-0", "The encoder embeds the query.")])
    hits = index.search("cross-encoder", top_k=3)
    assert hits[0].chunk_id == "hyphenated-0"
