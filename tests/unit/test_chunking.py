#!/usr/bin/env python3
"""
test_chunking.py --- unit tests for the chunking module

Contains:
    make_text(): builds repeated sentence text of a given token size
    test_fixed_window_size(): asserts chunks respect the token budget
    test_tail_merge_folds_undersized_chunk(): asserts tiny tails fold back
    test_overlap_windows_share_tokens(): asserts consecutive windows overlap
    test_chunk_ids_are_sequential(): asserts chunk ids number in order
    test_count_tokens(): asserts the token counter matches split length
"""

from src.ingest.chunking import ChunkConfig, TokenAwareChunker


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
    chunker = TokenAwareChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    assert all(chunk.token_count <= 24 for chunk in chunks)
    assert len(chunks) > 1


def test_overlap_windows_share_tokens() -> None:
    """Asserts consecutive windows share the overlap tokens."""
    chunker = TokenAwareChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    first_tail = chunks[0].text.split()[-6:]
    second_head = chunks[1].text.split()[:6]
    assert first_tail == second_head


def test_chunk_ids_are_sequential() -> None:
    """Asserts chunk ids number in document order."""
    chunker = TokenAwareChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_count_tokens() -> None:
    """Asserts the token counter matches a plain split."""
    chunker = TokenAwareChunker()
    assert chunker.count_tokens("one two three") == 3


def test_tail_merge_folds_undersized_chunk() -> None:
    """Asserts an undersized tail folds into the previous chunk."""
    chunker = TokenAwareChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    assert all(chunk.token_count >= 4 for chunk in chunks)


def test_chunk_ids_unique_per_document() -> None:
    """Asserts chunk ids are unique within a document."""
    chunker = TokenAwareChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split(make_text(30), "doc-1")
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_tiny_document_single_chunk() -> None:
    """Asserts a tiny document yields exactly one chunk."""
    chunker = TokenAwareChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    chunks = chunker.split("short text here", "doc-tiny")
    assert len(chunks) == 1


def test_empty_text_yields_no_chunks() -> None:
    """Asserts empty input yields no chunks."""
    chunker = TokenAwareChunker(ChunkConfig(max_tokens=24, overlap_tokens=6, min_chunk_tokens=4))
    assert chunker.split("", "doc-1") == []
    assert chunker.split("   ", "doc-1") == []
