#!/usr/bin/env python3
"""
test_prompts.py --- unit tests for the citation prompt templates

Contains:
    make_result(): builds a ranked result for prompt tests
    test_format_source_block_numbers_sources(): asserts source numbering
    test_prompt_contains_question_and_sources(): asserts prompt assembly
    test_prompt_limits_context_chunks(): asserts the context budget holds
"""

from src.generation.prompts.citation import build_citation_prompt, format_source_block
from src.retrieval.fusion import RankedResult


def make_result(index: int) -> RankedResult:
    """Builds a ranked result for prompt tests.

    Args:
        index: Index used for ids and text.

    Returns:
        result: Ranked result with clause text.
    """
    return RankedResult(
        chunk_id=f"doc-{index}-chunk-0",
        doc_id=f"doc-{index}",
        text=f"Section {index}.1 clause text.",
        score=0.9,
        source="fused",
    )


def test_format_source_block_numbers_sources() -> None:
    """Asserts source blocks are numbered from one."""
    assert format_source_block(make_result(1), 1) == "[1] Section 1.1 clause text."


def test_prompt_contains_question_and_sources() -> None:
    """Asserts the prompt carries the question and numbered sources."""
    prompt = build_citation_prompt("what does section 1.1 say?", [make_result(1)])
    assert "what does section 1.1 say?" in prompt
    assert "[1] Section 1.1 clause text." in prompt


def test_prompt_limits_context_chunks() -> None:
    """Asserts at most five sources make the prompt."""
    results = [make_result(index) for index in range(8)]
    prompt = build_citation_prompt("q?", results)
    assert "[6]" not in prompt


def test_prompt_starts_with_system_instructions() -> None:
    """Asserts the system prompt opens the assembled prompt."""
    prompt = build_citation_prompt("q?", [make_result(1)])
    assert prompt.startswith("You answer questions")


def test_source_block_includes_doc_label() -> None:
    """Asserts source blocks carry the document label."""
    assert "(doc: doc-1)" in format_source_block(make_result(1), 1)


def test_prompt_sources_numbered_sequentially() -> None:
    """Asserts sources number 1..n in order."""
    prompt = build_citation_prompt("q?", [make_result(1), make_result(2)])
    assert "[1]" in prompt and "[2]" in prompt


def test_system_prompt_forbids_outside_knowledge() -> None:
    """Asserts the system prompt bans outside knowledge."""
    from src.generation.prompts.citation import CITATION_SYSTEM_PROMPT

    assert "outside knowledge" in CITATION_SYSTEM_PROMPT


def test_prompt_instructs_cite_every_claim() -> None:
    """Asserts the prompt requires citing every claim."""
    from src.generation.prompts.citation import CITATION_SYSTEM_PROMPT

    assert "Every factual claim" in CITATION_SYSTEM_PROMPT


def test_refusal_guidance_present() -> None:
    """Asserts the refusal guidance survives rewording."""
    from src.generation.prompts.citation import CITATION_SYSTEM_PROMPT

    assert "do not know" in CITATION_SYSTEM_PROMPT
