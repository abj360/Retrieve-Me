#!/usr/bin/env python3
"""
chunking.py --- token-aware chunking for the ingestion pipeline

Contains:
    ChunkConfig: tunable token budget and overlap for the chunker
    Chunk: one chunk produced by the chunker
    Chunker: interface all chunkers implement
    SemanticClauseChunker: splits text on sentence and clause boundaries
"""

import re
from dataclasses import dataclass, field, replace
from typing import Protocol

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
CLAUSE_BOUNDARY = re.compile(
    r"(?<=\s)(?=(?:section|clause|article)\s+\d)", re.IGNORECASE
)
TOKEN_PATTERN = re.compile(r"\S+")  # whitespace-delimited approximation
DEFAULT_MAX_TOKENS = 512  # matches the embedder's comfortable context size
DEFAULT_OVERLAP_TOKENS = 64  # ~12% of the window
DEFAULT_MIN_CHUNK_TOKENS = 40  # smaller tails fold into the previous chunk


@dataclass(frozen=True)
class ChunkConfig:
    """Carries the token budget and overlap for the chunker.

    Attributes:
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Tokens shared between consecutive chunks.
        min_chunk_tokens: Minimum size before a chunk is merged away.
    """

    max_tokens: int = DEFAULT_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS


@dataclass(frozen=True)
class Chunk:
    """Carries one chunk produced by the chunker.

    Attributes:
        chunk_id: Stable identifier of the chunk.
        doc_id: Identifier of the document the chunk came from.
        text: Raw chunk text.
        token_count: Number of tokens in the chunk.
        index: Position of the chunk within the document.
        metadata: Extra attributes carried through to the indexes.
    """

    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    index: int
    metadata: dict = field(default_factory=dict)


class Chunker(Protocol):
    """Splits document text into chunks for indexing."""

    def split(self, text: str, doc_id: str) -> list[Chunk]:
        """Splits text into chunks for one document.

        Args:
            text: Raw document text.
            doc_id: Identifier of the document being split.

        Returns:
            chunks: Chunks covering the document in order.
        """


class SemanticClauseChunker:
    """Splits text on sentence and clause boundaries, packing to the token budget.

    Attributes:
        config: Token budget and overlap settings.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        """Stores the chunking configuration.

        Args:
            config: Token budget and overlap; defaults applied when omitted.
        """
        self.config = config or ChunkConfig()

    def count_tokens(self, text: str) -> int:
        """Counts whitespace-separated tokens in text.

        Args:
            text: Raw text to count.

        Returns:
            count: Number of tokens found.
        """
        return len(TOKEN_PATTERN.findall(text))

    def split(self, text: str, doc_id: str) -> list[Chunk]:
        """Splits text into chunks on sentence and clause boundaries.

        Args:
            text: Raw document text.
            doc_id: Identifier of the document being split.

        Returns:
            chunks: Boundary-aligned chunks covering the document in order.
        """
        sentences = self._split_sentences(text)
        return self._merge_tail(self._pack(sentences, doc_id))

    def _split_sentences(self, text: str) -> list[str]:
        """Splits raw text into sentences, breaking at clause markers.

        Args:
            text: Raw document text.

        Returns:
            sentences: Sentence and clause fragments in document order.
        """
        normalized = re.sub(r"\s*\n\s*", " ", text.strip())
        if not normalized:
            return []
        pieces: list[str] = []
        for sentence in SENTENCE_BOUNDARY.split(normalized):
            start = 0
            for match in CLAUSE_BOUNDARY.finditer(sentence):
                if match.start() > start:
                    pieces.append(sentence[start : match.start()].strip())
                    start = match.start()
            pieces.append(sentence[start:].strip())
        return [piece for piece in pieces if piece]

    def _pack(self, sentences: list[str], doc_id: str) -> list[Chunk]:
        """Packs sentences into chunks up to the token budget.

        Args:
            sentences: Sentence fragments in document order.
            doc_id: Identifier of the document being split.

        Returns:
            chunks: Packed chunks in document order.
        """
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            if current and current_tokens + sentence_tokens > self.config.max_tokens:
                chunks.append(self._make_chunk(current, doc_id, len(chunks)))
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += sentence_tokens
        if current:
            chunks.append(self._make_chunk(current, doc_id, len(chunks)))
        return chunks

    def _make_chunk(self, sentences: list[str], doc_id: str, index: int) -> Chunk:
        """Builds one chunk from packed sentences.

        Args:
            sentences: Sentences belonging to the chunk.
            doc_id: Identifier of the document being split.
            index: Position of the chunk within the document.

        Returns:
            chunk: Chunk covering the given sentences.
        """
        text = " ".join(sentences)
        return Chunk(
            chunk_id=f"{doc_id}-chunk-{index}",
            doc_id=doc_id,
            text=text,
            token_count=self.count_tokens(text),
            index=index,
        )

    def _merge_tail(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merges a tiny final chunk into its predecessor.

        Args:
            chunks: Chunks produced by the packer.

        Returns:
            merged: Chunks with an undersized tail folded back.
        """
        if len(chunks) > 1 and chunks[-1].token_count < self.config.min_chunk_tokens:
            tail = chunks.pop()
            previous = chunks[-1]
            merged_text = previous.text + " " + tail.text
            chunks[-1] = replace(
                previous, text=merged_text, token_count=self.count_tokens(merged_text)
            )
        return chunks
