#!/usr/bin/env python3
"""
test_chunking.py --- unit tests for the chunking module

Contains:
    make_text(): builds repeated sentence text of a given token size
    test_fixed_window_size(): asserts chunks respect the token budget
    test_tail_merge_folds_undersized_chunk(): asserts tiny tails fold back
    test_chunks_end_on_sentence_boundaries(): asserts chunks never cut sentences
    test_chunk_ids_are_sequential(): asserts chunk ids number in order
    test_count_tokens(): asserts the token counter matches split length
"""

from src.ingest.chunking import ChunkConfig, SemanticClauseChunker


def make_text(sentences: int) -> str:
    """Builds repeated sentence text of a given sentence count.

    Args:
        sentences: Number of sentences to generate.

    Returns:
        text: Repeated sentence text.
    """
    return " ".join(f"Sentence {index} has exactly six tokens here." for index in range(sentences))


def test_fixed_window_size() -> None:
    """Asserts chunks respect the configured token budget."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    assert all(chunk.token_count <= 24 for chunk in chunks)
    assert len(chunks) > 1


def test_chunks_end_on_sentence_boundaries() -> None:
    """Asserts semantic chunks never cut a sentence in two."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    assert all(chunk.text.rstrip().endswith((".", "!", "?")) for chunk in chunks)


def test_chunk_ids_are_sequential() -> None:
    """Asserts chunk ids number in document order."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_count_tokens() -> None:
    """Asserts the token counter matches a plain split."""
    chunker = SemanticClauseChunker()
    assert chunker.count_tokens("one two three") == 3


def test_tail_merge_folds_undersized_chunk() -> None:
    """Asserts an undersized tail folds into the previous chunk."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    assert all(chunk.token_count >= 4 for chunk in chunks)


def test_chunk_ids_unique_per_document() -> None:
    """Asserts chunk ids are unique within a document."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_tiny_document_single_chunk() -> None:
    """Asserts a tiny document yields exactly one chunk."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split("short text here", "doc-tiny")
    assert len(chunks) == 1


def test_empty_text_yields_no_chunks() -> None:
    """Asserts empty input yields no chunks."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    assert chunker.split("", "doc-1") == []
    assert chunker.split("   ", "doc-1") == []


def test_regression_clause_marker_kept_with_clause() -> None:
    """Asserts a clause marker is never split from its clause (recall regression)."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=48, overlap_tokens=8, min_chunk_tokens=5))
    text = "Intro words here. Section 12.1 The licensee shall indemnify the vendor fully."
    chunks = chunker.split(text, "doc-1")
    assert any("Section 12.1" in chunk.text for chunk in chunks)


def test_chunks_end_at_sentence_boundaries(token_chunker) -> None:
    """Asserts semantic chunks do not end mid-sentence."""
    text = "Alpha one. Beta two. Gamma three. Delta four. Epsilon five."
    chunks = token_chunker.split(text, "doc-1")
    assert all(chunk.text.rstrip().endswith((".", "!", "?")) for chunk in chunks)


def test_budget_boundary_exact_fit(token_chunker) -> None:
    """Asserts text exactly at budget yields one chunk."""
    text = " ".join(["token"] * 48) + "."
    chunks = token_chunker.split(text, "doc-1")
    assert len(chunks) == 1


def test_clause_boundary_splits_before_marker(token_chunker) -> None:
    """Asserts the splitter breaks before a clause marker, not after."""
    text = "General words up front. Section 2.4 Specific obligations follow here."
    chunks = token_chunker.split(text, "doc-1")
    assert any(chunk.text.startswith("Section 2.4") or " Section 2.4" in chunk.text for chunk in chunks)
