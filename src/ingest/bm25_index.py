#!/usr/bin/env python3
"""
bm25_index.py --- BM25 sparse index over chunk text using rank_bm25

Contains:
    tokenize(): splits text into lowercase search terms
    BM25Hit: one scored hit from the sparse index
    BM25Index: builds and searches the sparse index
    BM25Index.save(): pickles the built index to disk
    BM25Index.load(): restores a pickled index from disk
"""

import logging
import pickle
import re
from pathlib import Path
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Splits text into lowercase search terms.

    Args:
        text: Raw chunk or query text.

    Returns:
        terms: Lowercase tokens; hyphenated terms stay whole.
    """
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Hit:
    """Carries one scored hit from the sparse index.

    Attributes:
        chunk_id: Stable identifier of the matched chunk.
        doc_id: Identifier of the document the chunk came from.
        text: Raw chunk text.
        score: BM25 score assigned by the index.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float


class BM25Index:
    """Builds and searches a BM25 sparse index over chunk text; hyphens stay whole.

    Attributes:
        chunk_ids: Identifiers of the indexed chunks, in build order.
    """

    def __init__(self) -> None:
        """Creates an empty index; build() must be called before search."""
        self.chunk_ids: list[str] = []
        self._chunks: list = []
        self._index: BM25Okapi | None = None

    def build(self, chunks: list) -> int:
        """Builds the sparse index from chunk objects.

        Args:
            chunks: Chunks exposing chunk_id, doc_id, and text.

        Returns:
            indexed: Number of chunks indexed.
        """
        logger.info("building BM25 index over %d chunks", len(chunks))
        self._chunks = list(chunks)
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
        return len(self.chunk_ids)

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        """Searches the index for chunks matching the query.

        Args:
            query: Raw query text.
            top_k: Maximum number of hits to return.

        Returns:
            hits: Scored sparse hits with chunk text, best first.
        """
        if self._index is None:
            raise RuntimeError("BM25Index.search called before build()")
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [
            BM25Hit(
                chunk_id=self._chunks[position].chunk_id,
                doc_id=self._chunks[position].doc_id,
                text=self._chunks[position].text,
                score=score,
            )
            for position, score in ranked[:top_k]
        ]

    def save(self, path: Path) -> None:
        """Pickles the built index to disk.

        Args:
            path: Destination file for the pickled index.
        """
        if self._index is None:
            raise RuntimeError("BM25Index.save called before build()")
        state = {"chunk_ids": self.chunk_ids, "chunks": self._chunks, "index": self._index}
        with path.open("wb") as handle:
            pickle.dump(state, handle)

    def load(self, path: Path) -> None:
        """Restores a pickled index from disk.

        Args:
            path: File written by an earlier save() call.
        """
        with path.open("rb") as handle:
            state = pickle.load(handle)
        self.chunk_ids = state["chunk_ids"]
        self._chunks = state["chunks"]
        self._index = state["index"]
