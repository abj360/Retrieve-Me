#!/usr/bin/env python3
"""
generate.py --- citation-grounded generation over retrieved chunks

Contains:
    Citation: one grounded citation back to a retrieved chunk
    GeneratedAnswer: an answer with its grounded citations
    CitationGenerator: generates citation-grounded answers
"""

import logging
import re
from dataclasses import dataclass

from src.retrieval.fusion import RankedResult

logger = logging.getLogger(__name__)

CITATION_SYSTEM_PROMPT = """You answer questions using only the provided sources.
Cite every claim with the source number in square brackets, like [1] or [2].
If the sources do not contain the answer, say you do not know."""

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
MAX_CONTEXT_CHUNKS = 5


@dataclass(frozen=True)
class Citation:
    """Carries one grounded citation back to a retrieved chunk.

    Attributes:
        chunk_id: Chunk the citation points at.
        doc_id: Document the chunk came from.
        quote: Short quoted span supporting the claim.
    """

    chunk_id: str
    doc_id: str
    quote: str


@dataclass(frozen=True)
class GeneratedAnswer:
    """Carries an answer with its grounded citations.

    Attributes:
        answer: Generated answer text with [n] citation markers.
        citations: Grounded citations referenced by the markers.
    """

    answer: str
    citations: list[Citation]


class CitationGenerator:
    """Generates citation-grounded answers over retrieved chunks.

    Attributes:
        max_tokens: Maximum tokens per generated answer.
    """

    def __init__(self, llm_client, max_tokens: int = 512) -> None:
        """Stores the LLM client and token budget.

        Args:
            llm_client: Callable returning completion text for a prompt.
            max_tokens: Maximum tokens per generated answer.
        """
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    def generate(self, query: str, results: list[RankedResult]) -> GeneratedAnswer:
        """Generates a citation-grounded answer for a query.

        Args:
            query: Raw query text.
            results: Retrieved chunks to ground the answer on.

        Returns:
            answer: Generated answer with validated citations.
        """
        prompt = self._build_prompt(query, results)
        raw = self.llm_client(prompt, max_tokens=self.max_tokens)
        return GeneratedAnswer(answer=raw, citations=self._parse_citations(raw, results))

    def _build_prompt(self, query: str, results: list[RankedResult]) -> str:
        """Builds the grounded-generation prompt.

        Args:
            query: Raw query text.
            results: Retrieved chunks to ground on.

        Returns:
            prompt: System prompt plus numbered sources and the question.
        """
        sources = "\n\n".join(
            f"[{index}] {result.text}" for index, result in enumerate(results[:MAX_CONTEXT_CHUNKS], 1)
        )
        return f"{CITATION_SYSTEM_PROMPT}\n\nSources:\n{sources}\n\nQuestion: {query}"

    def _parse_citations(self, answer: str, results: list[RankedResult]) -> list[Citation]:
        """Parses [n] citation markers into grounded citations.

        Args:
            answer: Generated answer text.
            results: Retrieved chunks the markers reference.

        Returns:
            citations: Citations in order of first appearance.
        """
        citations: list[Citation] = []
        for match in CITATION_PATTERN.finditer(answer):
            index = int(match.group(1)) - 1
            if index >= len(results):
                continue
            result = results[index]
            citations.append(
                Citation(chunk_id=result.chunk_id, doc_id=result.doc_id, quote=result.text[:80])
            )
        return citations
