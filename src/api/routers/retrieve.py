#!/usr/bin/env python3
"""
retrieve.py --- POST /retrieve endpoint for the retrieval pipeline

Contains:
    RetrieveRequest: incoming retrieval request payload
    RetrievedChunk: one scored chunk in the response
    RetrieveResponse: retrieval response payload
    cache_key(): builds the canonical cache key for a request
    paginate(): slices one page out of the ranked match list
    retrieve(): runs retrieval for a query and returns scored chunks
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.cache import QueryCache
from src.api.dependencies import get_pipeline, get_query_cache
from src.retrieval.strategies import HybridRetriever

logger = logging.getLogger("retrieval.retrieve")

router = APIRouter(tags=["retrieval"])
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

    query: str = Field(min_length=1, description="Query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Max chunks returned")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
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
        has_next: Whether a further page of results exists.
        took_ms: Wall-clock time spent serving the request.
    """

    query: str
    results: list[RetrievedChunk]
    total: int
    page: int
    page_size: int
    applied_filters: dict[str, str] | None
    has_next: bool
    took_ms: float


def cache_key(payload: RetrieveRequest) -> str:
    """Builds the canonical cache key for one retrieval request.

    Args:
        payload: Request the key should identify.

    Returns:
        key: Opaque cache key covering query, pagination, and filters.
    """
    filters = tuple(sorted((payload.filters or {}).items()))
    return f"q:{payload.query}|k:{payload.top_k}|p:{payload.page}|s:{payload.page_size}|f:{filters}"


def paginate(
    matches: list[RetrievedChunk], page: int, page_size: int
) -> list[RetrievedChunk]:
    """Slices one page out of the full match list.

    Args:
        matches: All matched chunks in rank order.
        page: One-based page number to return.
        page_size: Number of results per page.

    Returns:
        page_results: The slice of matches belonging to the requested page.
    """
    start = (page - 1) * page_size
    end = start + page_size
    return matches[start:end]


@router.post("/retrieve", response_model=RetrieveResponse, summary="Hybrid retrieval search")
def retrieve(
    payload: RetrieveRequest,
    cache: QueryCache = Depends(get_query_cache),
    pipeline: HybridRetriever = Depends(get_pipeline),
) -> RetrieveResponse:
    """Returns retrieval results for one query.

    Args:
        payload: Retrieval request with query, top_k, and pagination.
        cache: Query-response cache injected by FastAPI.
        pipeline: Hybrid retrieval pipeline injected by FastAPI.

    Returns:
        response: One page of scored chunks plus pagination metadata.
    """
    started = time.perf_counter()
    key = cache_key(payload)
    cached = cache.get(key)
    if cached is not None:
        logger.debug("cache hit for %s", key)
        return cached
    logger.info("retrieve q=%r top_k=%d page=%d", payload.query, payload.top_k, payload.page)
    try:
        ranked = pipeline.retrieve(payload.query, top_k=payload.top_k, filters=payload.filters)
    except Exception as exc:
        logger.exception("pipeline failed for query %r", payload.query)
        raise HTTPException(status_code=502, detail="retrieval backend unavailable") from exc
    matches = [
        RetrievedChunk(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            text=hit.text,
            score=hit.score,
            source=hit.source,
        )
        for hit in ranked
    ]
    page_results = paginate(matches, payload.page, payload.page_size)
    response = RetrieveResponse(
        query=payload.query,
        results=page_results,
        total=len(matches),
        page=payload.page,
        page_size=payload.page_size,
        applied_filters=payload.filters,
        has_next=payload.page * payload.page_size < len(matches),
        took_ms=(time.perf_counter() - started) * 1000,
    )
    cache.set(key, response)
    return response
