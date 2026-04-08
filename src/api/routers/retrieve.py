#!/usr/bin/env python3
"""
retrieve.py --- POST /retrieve endpoint for the retrieval pipeline

Contains:
    RetrieveRequest: incoming retrieval request payload
    RetrievedChunk: one scored chunk in the response
    RetrieveResponse: retrieval response payload
    stub_results(): builds deterministic stub results while indexing lands
    retrieve(): runs retrieval for a query and returns scored chunks
"""

import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger("retrieval.retrieve")

router = APIRouter()

STUB_RESULT_COUNT = 30
DEFAULT_PAGE_SIZE = 10


class RetrieveRequest(BaseModel):
    """Carries one retrieval request.

    Attributes:
        query: Raw query text from the caller.
        top_k: Maximum number of chunks to return.
        page: One-based results page to return (default 1).
        page_size: Number of results per page.
    """

    query: str
    top_k: int = 10
    page: int = 1
    page_size: int = 10


class RetrievedChunk(BaseModel):
    """Carries one scored chunk in a retrieval response.

    Attributes:
        chunk_id: Stable identifier of the chunk.
        doc_id: Identifier of the document the chunk came from.
        text: Raw chunk text.
        score: Relevance score assigned by the pipeline.
        source: Retrieval stage that produced the chunk.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str


class RetrieveResponse(BaseModel):
    """Carries the scored chunk list back to the caller.

    Attributes:
        query: Query text the results were produced for.
        results: Scored chunks for the requested page, best first.
        total: Total number of matched chunks across all pages.
        page: One-based page number being returned.
        page_size: Number of results per page.
        took_ms: Wall-clock time spent serving the request.
    """

    query: str
    results: list[RetrievedChunk]
    total: int
    page: int
    page_size: int
    took_ms: float


def stub_results(query: str) -> list[RetrievedChunk]:
    """Builds deterministic stub results until the index wiring lands.

    Args:
        query: Query text to echo into the stub passages.

    Returns:
        results: Thirty stub chunks with decreasing scores.
    """
    return [
        RetrievedChunk(
            chunk_id=f"stub-{index}",
            doc_id=f"stub-doc-{index}",
            text=f"stub passage {index} for query {query!r}",
            score=1.0 / (index + 1),
            source="stub",
        )
        for index in range(STUB_RESULT_COUNT)
    ]


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    """Returns retrieval results for one query.

    Args:
        payload: Retrieval request with query, top_k, and pagination.

    Returns:
        response: One page of scored chunks plus pagination metadata.
    """
    started = time.perf_counter()
    logger.info("retrieve called with top_k=%d page=%d", payload.top_k, payload.page)
    matches = stub_results(payload.query)[: payload.top_k]
    start = (payload.page - 1) * payload.page_size
    page_results = matches[start : start + payload.page_size]
    return RetrieveResponse(
        query=payload.query,
        results=page_results,
        total=len(matches),
        page=payload.page,
        page_size=payload.page_size,
        took_ms=(time.perf_counter() - started) * 1000,
    )
