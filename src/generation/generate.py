#!/usr/bin/env python3
"""
generate.py --- citation-grounded generation over retrieved chunks (with retry)

Contains:
    Citation: one grounded citation back to a retrieved chunk (chunk, doc, quote)
    GeneratedAnswer: an answer with its grounded citations
    CitationGenerator: generates citation-grounded answers
    CitationGenerator.faithfulness(): scores citation grounding against results

Prompt templates live in src/generation/prompts/citation.py.
"""

import logging
import re
from dataclasses import dataclass

from src.generation.prompts.citation import build_citation_prompt
from src.retrieval.fusion import RankedResult

logger = logging.getLogger(__name__)

# matches the [n] markers the citation prompt asks the model to emit
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


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
        raw = self._call_llm(prompt)
        return GeneratedAnswer(answer=raw, citations=self._parse_citations(raw, results))

    def _build_prompt(self, query: str, results: list[RankedResult]) -> str:
        """Builds the grounded-generation prompt.

        Args:
            query: Raw query text.
            results: Retrieved chunks to ground on.

        Returns:
            prompt: System prompt plus numbered sources and the question.
        """
        return build_citation_prompt(query, results)

    def _parse_citations(self, answer: str, results: list[RankedResult]) -> list[Citation]:
        """Parses [n] citation markers into grounded citations.

        Args:
            answer: Generated answer text.
            results: Retrieved chunks the markers reference.

        Returns:
            citations: Grounded, deduplicated citations in order of appearance.
        """
        citations: list[Citation] = []
        seen: set[int] = set()
        for match in CITATION_PATTERN.finditer(answer):
            index = int(match.group(1)) - 1
            if index < 0 or index >= len(results) or index in seen:
                continue
            seen.add(index)
            result = results[index]
            citations.append(
                Citation(chunk_id=result.chunk_id, doc_id=result.doc_id, quote=result.text[:80])
            )
        return citations


    def _call_llm(self, prompt: str) -> str:
        """Calls the LLM client with one retry on transient failure.

        Args:
            prompt: Assembled grounded-generation prompt.

        Returns:
            completion: Raw completion text.
        """
        try:
            return self.llm_client(prompt, max_tokens=self.max_tokens)
        except Exception as exc:
            logger.warning("generation call failed, one retry: %s", exc)
            return self.llm_client(prompt, max_tokens=self.max_tokens)


    def faithfulness(self, answer: GeneratedAnswer, results: list[RankedResult]) -> float:
        """Scores how grounded an answer's citations are.

        Args:
            answer: Generated answer to score.
            results: Retrieved chunks the answer should be grounded in.

        Returns:
            score: Fraction of citations that map to retrieved chunks.
        """
        from src.eval.metrics import citation_faithfulness

        return citation_faithfulness(
            [citation.chunk_id for citation in answer.citations],
            {result.chunk_id for result in results},
        )
