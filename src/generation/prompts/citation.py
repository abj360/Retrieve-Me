#!/usr/bin/env python3
"""
citation.py --- prompt templates for citation-grounded generation

Contains:
    CITATION_SYSTEM_PROMPT: system instructions for grounded answers
    build_citation_prompt(): assembles the grounded-generation prompt
    format_source_block(): formats one numbered source block
"""

from src.retrieval.fusion import RankedResult

CITATION_SYSTEM_PROMPT = """You answer questions using only the provided sources.
Cite every claim with the source number in square brackets, like [1] or [2].
If the sources do not contain the answer, say you do not know."""

MAX_CONTEXT_CHUNKS = 5


def format_source_block(result: RankedResult, index: int) -> str:
    """Formats one numbered source block for the prompt.

    Args:
        result: Retrieved chunk to present as a source.
        index: One-based source number used by citation markers.

    Returns:
        block: Numbered source block for the prompt.
    """
    return f"[{index}] (doc: {result.doc_id}) {result.text}"


def build_citation_prompt(query: str, results: list[RankedResult]) -> str:
    """Assembles the grounded-generation prompt.

    Args:
        query: Raw query text.
        results: Retrieved chunks to ground the answer on.

    Returns:
        prompt: System prompt plus numbered sources and the question.
    """
    sources = "\n\n".join(
        format_source_block(result, index)
        for index, result in enumerate(results[:MAX_CONTEXT_CHUNKS], 1)
    )
    return f"{CITATION_SYSTEM_PROMPT}\n\nSources:\n{sources}\n\nQuestion: {query}"
