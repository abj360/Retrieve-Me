#!/usr/bin/env python3
"""
retrieve.py --- POST /retrieve endpoint for the retrieval pipeline

Contains:
    RetrieveRequest: incoming retrieval request payload
    RetrievedChunk: one scored chunk in the response
    RetrieveResponse: retrieval response payload
    stub_results(): builds deterministic stub results while indexing lands
    cache_key(): builds the canonical cache key for a request
    retrieve(): runs retrieval for a query and returns scored chunks
"""

import logging
import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.cache import QueryCache
from src.api.dependencies import get_query_cache

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
        filters: Optional exact-match metadata filters on chunk payloads.
    """

    query: str
    top_k: int = 10
    page: int = 1
    page_size: int = 10
    filters: dict[str, str] | None = None


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
        applied_filters: Filters applied to the search, echoed back to the caller.
        took_ms: Wall-clock time spent serving the request.
    """

    query: str
    results: list[RetrievedChunk]
    total: int
    page: int
    page_size: int
    applied_filters: dict[str, str] | None
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


def cache_key(payload: RetrieveRequest) -> str:
    """Builds the canonical cache key for one retrieval request.

    Args:
        payload: Request the key should identify.

    Returns:
        key: Opaque cache key covering query, pagination, and filters.
    """
    filters = tuple(sorted((payload.filters or {}).items()))
    return f"q:{payload.query}|k:{payload.top_k}|p:{payload.page}|s:{payload.page_size}|f:{filters}"


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    payload: RetrieveRequest,
    cache: QueryCache = Depends(get_query_cache),
) -> RetrieveResponse:
    """Returns retrieval results for one query.

    Args:
        payload: Retrieval request with query, top_k, and pagination.

    Returns:
        response: One page of scored chunks plus pagination metadata.
    """
    started = time.perf_counter()
    key = cache_key(payload)
    cached = cache.get(key)
    if cached is not None:
        logger.debug("cache hit for %s", key)
        return cached
    logger.info("retrieve called with top_k=%d page=%d", payload.top_k, payload.page)
    matches = stub_results(payload.query)[: payload.top_k]
    start = (payload.page - 1) * payload.page_size
    page_results = matches[start : start + payload.page_size]
    response = RetrieveResponse(
        query=payload.query,
        results=page_results,
        total=len(matches),
        page=payload.page,
        page_size=payload.page_size,
        applied_filters=payload.filters,
        took_ms=(time.perf_counter() - started) * 1000,
    )
    cache.set(key, response)
    return response
