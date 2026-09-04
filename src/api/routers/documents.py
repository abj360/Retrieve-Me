#!/usr/bin/env python3
"""
documents.py --- POST /documents endpoint for uploading a corpus from the browser

Contains:
    UploadedDocument: one accepted file and what it produced
    UploadResponse: upload response payload
    MAX_UPLOAD_BYTES: per-file ceiling accepted from a browser
    decode(): reads an uploaded file as text, rejecting anything undecodable
    upload_documents(): chunks, embeds, and indexes uploaded files
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from src.ingest.loader import TEXT_SUFFIXES, CorpusIngestor, Document, build_ingestor

logger = logging.getLogger("retrieval.documents")

router = APIRouter(tags=["documents"], prefix="")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class UploadedDocument(BaseModel):
    """Reports what one accepted file contributed to the index.

    Attributes:
        doc_id: Identifier the document was indexed under.
        title: Original file name.
        bytes: Size of the uploaded file.
    """

    doc_id: str
    title: str
    bytes: int


class UploadResponse(BaseModel):
    """Reports the outcome of one upload.

    Attributes:
        documents: The files that were indexed.
        chunks: Chunks written to the sparse and dense indexes.
        skipped: File names rejected, each with the reason.
        took_ms: Wall-clock time the ingestion took.
    """

    documents: list[UploadedDocument]
    chunks: int
    skipped: dict[str, str]
    took_ms: int


def decode(raw: bytes, name: str) -> str:
    """Reads an uploaded file as UTF-8 text.

    Args:
        raw: Bytes as uploaded.
        name: File name, used in the error message.

    Returns:
        text: Decoded document text.

    Raises:
        ValueError: When the bytes are not valid UTF-8 text.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8 text") from error


@router.post("/documents", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile],
    ingestor: CorpusIngestor = Depends(build_ingestor),
) -> UploadResponse:
    """Indexes uploaded documents so they are searchable immediately.

    Ingestion is the one part of the pipeline that had no route: a corpus could
    only be loaded from disk by the CLI, which put it out of reach of anyone
    without shell access to the container. Unreadable files are reported rather
    than failing the whole upload, so one bad file does not lose the batch.

    Args:
        files: Uploaded files; .txt and .md are indexed, anything else is skipped.
        ingestor: Ingestor wired to the live sparse and dense stores.

    Returns:
        response: What was indexed, what was skipped, and how long it took.

    Raises:
        HTTPException: 400 when the upload carries no indexable file.
    """
    started = time.perf_counter()
    documents: list[Document] = []
    accepted: list[UploadedDocument] = []
    skipped: dict[str, str] = {}

    for upload in files:
        name = upload.filename or "unnamed"
        suffix = name[name.rfind(".") :].lower() if "." in name else ""
        if suffix not in TEXT_SUFFIXES:
            skipped[name] = f"unsupported type {suffix or 'none'}"
            continue
        raw = await upload.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            skipped[name] = f"larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB"
            continue
        try:
            text = decode(raw, name)
        except ValueError as error:
            skipped[name] = str(error)
            continue
        doc_id = name[: name.rfind(".")] if "." in name else name
        documents.append(Document(doc_id=doc_id, title=name, text=text))
        accepted.append(UploadedDocument(doc_id=doc_id, title=name, bytes=len(raw)))

    if not documents:
        raise HTTPException(status_code=400, detail=f"no indexable documents: {skipped}")

    stats = ingestor.ingest(documents)
    logger.info("indexed %d uploaded documents into %d chunks", stats.documents, stats.chunks)
    return UploadResponse(
        documents=accepted,
        chunks=stats.chunks,
        skipped=skipped,
        took_ms=int((time.perf_counter() - started) * 1000),
    )
