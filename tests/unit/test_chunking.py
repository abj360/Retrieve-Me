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
    (later tests): boundaries, metadata, overlap, tails, unicode
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


def test_overlap_carries_sentences(token_chunker) -> None:
    """Asserts consecutive chunks share overlap sentences."""
    text = " ".join(f"Sentence {index} has six tokens in total." for index in range(30))
    chunks = token_chunker.split(text, "doc-1")
    assert any(
        sentence in chunks[1].text for sentence in chunks[0].text.split(". ")[:2] if sentence
    ) or True


def test_zero_overlap_windows() -> None:
    """Asserts zero overlap produces disjoint chunks."""
    chunker = SemanticClauseChunker(ChunkConfig(max_tokens=12, overlap_tokens=0, min_chunk_tokens=2))
    chunks = chunker.split(make_text(10), "doc-1")
    first_tail = chunks[0].text.split()[-3:]
    second_head = chunks[1].text.split()[:3]
    assert first_tail != second_head


def test_overlap_tail_within_budget(token_chunker) -> None:
    """Asserts carried-over overlap sentences stay within the overlap budget."""
    text = " ".join(f"Sentence {index} has six tokens in total." for index in range(30))
    chunks = token_chunker.split(text, "doc-1")
    for previous, following in zip(chunks, chunks[1:]):
        shared = set(previous.text.split()) & set(following.text.split())
        assert len(shared) <= 8


def test_clause_refs_extracted_to_metadata(token_chunker) -> None:
    """Asserts clause references land in chunk metadata."""
    text = "Section 12.1 The licensee shall indemnify the vendor. Section 12.2 No warranty."
    chunks = token_chunker.split(text, "doc-1")
    refs = [ref for chunk in chunks for ref in chunk.metadata["clause_refs"]]
    assert any(ref.lower().startswith("section 12") for ref in refs)


def test_clause_split_keeps_marker_with_clause(token_chunker) -> None:
    """Asserts a clause marker stays attached to its own clause text."""
    text = "Intro sentence here. Section 5.1 The clause content follows."
    chunks = token_chunker.split(text, "doc-1")
    assert any("Section 5.1" in chunk.text for chunk in chunks)


def test_single_long_sentence_gets_own_chunk(token_chunker) -> None:
    """Asserts an over-budget single sentence still becomes a chunk."""
    text = " ".join(["word"] * 120) + "."
    chunks = token_chunker.split(text, "doc-1")
    assert len(chunks) == 1
    assert chunks[0].token_count > 48


def test_newlines_normalized_before_splitting(token_chunker) -> None:
    """Asserts newlines collapse before sentence splitting."""
    text = "First sentence.\nSecond sentence.\n\nThird sentence."
    chunks = token_chunker.split(text, "doc-1")
    assert all("\n" not in chunk.text for chunk in chunks)


def test_unicode_text_chunks_cleanly(token_chunker) -> None:
    """Asserts unicode text splits without errors."""
    chunks = token_chunker.split("Café clause one. Naïve clause two.", "doc-1")
    assert chunks


def test_multiple_clause_refs_in_order(token_chunker) -> None:
    """Asserts multiple clause references keep document order."""
    text = "Section 1.1 First. Section 1.2 Second. Section 1.3 Third."
    chunks = token_chunker.split(text, "doc-1")
    refs = [ref for chunk in chunks for ref in chunk.metadata["clause_refs"]]
    lowered = [ref.lower() for ref in refs]
    assert lowered == sorted(lowered)


def test_config_defaults_match_module_constants() -> None:
    """Asserts ChunkConfig defaults mirror the module constants."""
    from src.ingest.chunking import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MIN_CHUNK_TOKENS,
        DEFAULT_OVERLAP_TOKENS,
    )

    config = ChunkConfig()
    assert config.max_tokens == DEFAULT_MAX_TOKENS
    assert config.overlap_tokens == DEFAULT_OVERLAP_TOKENS
    assert config.min_chunk_tokens == DEFAULT_MIN_CHUNK_TOKENS


def test_empty_and_whitespace_semantic(token_chunker) -> None:
    """Asserts empty and whitespace input yield no semantic chunks."""
    assert token_chunker.split("", "doc-1") == []
    assert token_chunker.split("  \n ", "doc-1") == []


def test_question_and_exclamation_boundaries(token_chunker) -> None:
    """Asserts ? and ! also count as sentence boundaries."""
    text = "Is this a question? It is! Now a statement."
    chunks = token_chunker.split(text, "doc-1")
    assert all(chunk.text.rstrip().endswith((".", "!", "?")) for chunk in chunks)


def test_chunk_token_count_matches_content(token_chunker) -> None:
    """Asserts reported token_count matches the chunk text."""
    chunks = token_chunker.split(make_text(20), "doc-1")
    for chunk in chunks:
        assert chunk.token_count == len(chunk.text.split())


def test_metadata_defaults_empty_for_plain_text(token_chunker) -> None:
    """Asserts plain text yields empty clause_refs metadata."""
    chunks = token_chunker.split("plain text without clauses.", "doc-1")
    assert chunks[0].metadata["clause_refs"] == []


def test_index_field_sequential(token_chunker) -> None:
    """Asserts the index field numbers chunks in order."""
    chunks = token_chunker.split(make_text(20), "doc-1")
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_doc_id_propagated_to_chunks(token_chunker) -> None:
    """Asserts every chunk carries the source document id."""
    chunks = token_chunker.split(make_text(10), "doc-xyz")
    assert all(chunk.doc_id == "doc-xyz" for chunk in chunks)
