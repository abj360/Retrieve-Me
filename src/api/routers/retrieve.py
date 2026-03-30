#!/usr/bin/env python3
"""
retrieve.py --- POST /retrieve endpoint for the retrieval pipeline

Contains:
    RetrieveRequest: incoming retrieval request payload
    RetrievedChunk: one scored chunk in the response
    RetrieveResponse: retrieval response payload
    retrieve(): runs retrieval for a query and returns scored chunks
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("retrieval.retrieve")

router = APIRouter()

STUB_RESULT_COUNT = 30


class RetrieveRequest(BaseModel):
    """Carries one retrieval request.

    Attributes:
        query: Raw query text from the caller.
        top_k: Maximum number of chunks to return.
    """

    query: str
    top_k: int = 10


class RetrievedChunk(BaseModel):
    """Carries one scored chunk in a retrieval response.

    Attributes:
        chunk_id: Stable identifier of the chunk.
        doc_id: Identifier of the document the chunk came from.
        text: Raw chunk text.
        score: Relevance score assigned by the pipeline.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float


class RetrieveResponse(BaseModel):
    """Carries the scored chunk list back to the caller.

    Attributes:
        query: Query text the results were produced for.
        results: Scored chunks, best first.
        took_ms: Wall-clock time spent serving the request.
    """

    query: str
    results: list[RetrievedChunk]
    took_ms: float


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    """Returns retrieval results for one query.

    Args:
        payload: Retrieval request with the query and top_k.

    Returns:
        response: Scored chunks and timing for the request.
    """
    logger.info("retrieve called with top_k=%d", payload.top_k)
    results = [
        RetrievedChunk(
            chunk_id=f"stub-{index}",
            doc_id=f"stub-doc-{index}",
            text=f"stub passage {index} matching {payload.query!r}",
            score=1.0 / (index + 1),
        )
        for index in range(STUB_RESULT_COUNT)
    ]
    return RetrieveResponse(query=payload.query, results=results[: payload.top_k], took_ms=1.0)
