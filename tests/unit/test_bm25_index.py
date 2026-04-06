#!/usr/bin/env python3
"""
test_bm25_index.py --- unit tests for the BM25 sparse index

Contains:
    make_chunk(): builds a minimal chunk-like object for indexing
    test_tokenize_lowercases_and_keeps_hyphens(): asserts tokenizer behavior
    test_tokenize_lowercases_and_keeps_hyphens(): asserts tokenizer behavior
    test_search_ranks_relevant_chunk_first(): asserts bm25 ranks exact terms well
    test_search_before_build_raises(): asserts the index must be built first
"""

from types import SimpleNamespace

import pytest

from src.ingest.bm25_index import BM25Index, tokenize


def make_chunk(chunk_id: str, text: str) -> SimpleNamespace:
    """Builds a minimal chunk-like object for indexing.

    Args:
        chunk_id: Identifier for the chunk.
        text: Chunk text to index.

    Returns:
        chunk: Object exposing chunk_id, doc_id, and text.
    """
    return SimpleNamespace(chunk_id=chunk_id, doc_id=chunk_id.split("-")[0], text=text)


def test_tokenize_lowercases_and_keeps_hyphens() -> None:
    """Asserts the tokenizer lowercases and keeps hyphenated terms whole."""
    assert tokenize("Multi-Agent SYSTEMS, Inc.") == ["multi-agent", "systems", "inc"]


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
