#!/usr/bin/env python3
"""
chunking.py --- token-aware chunking for the ingestion pipeline

Contains:
    ChunkConfig: tunable token budget and overlap for the chunker
    Chunk: one chunk produced by the chunker
    Chunker: interface all chunkers implement
    TokenAwareChunker: splits text into fixed-size token windows with overlap
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

TOKEN_PATTERN = re.compile(r"\S+")
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_MIN_CHUNK_TOKENS = 40


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


class TokenAwareChunker:
    """Splits text into fixed-size token windows with overlap.

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
        """Splits text into fixed-size token windows with overlap.

        Args:
            text: Raw document text.
            doc_id: Identifier of the document being split.

        Returns:
            chunks: Fixed-size chunks covering the document in order.
        """
        tokens = TOKEN_PATTERN.findall(text)
        if not tokens:
            return []
        step = self.config.max_tokens - self.config.overlap_tokens
        chunks = []
        for index, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + self.config.max_tokens]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-chunk-{index}",
                    doc_id=doc_id,
                    text=" ".join(window),
                    token_count=len(window),
                    index=index,
                )
            )
            if start + self.config.max_tokens >= len(tokens):
                break
        return chunks
