#!/usr/bin/env python3
"""
test_documents_upload.py --- unit tests for the document upload endpoint

Contains:
    RecordingIngestor: ingestor double capturing the documents handed to it
    client(): TestClient with the ingestor dependency replaced
    test_indexes_an_uploaded_document(): a .txt file reaches the ingestor
    test_reports_chunks_written(): the response carries what was indexed
    test_skips_an_unsupported_type(): a .pdf is reported, not indexed
    test_skips_a_file_that_is_not_utf8(): undecodable bytes are reported
    test_one_bad_file_does_not_lose_the_batch(): the good file still indexes
    test_rejects_an_upload_with_nothing_indexable(): answers 400
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.ingest.loader import Document, IngestStats, build_ingestor


class RecordingIngestor:
    """Ingestor double that records what it was asked to index.

    Attributes:
        documents: Documents passed to ingest().
    """

    def __init__(self) -> None:
        """Starts with nothing recorded."""
        self.documents: list[Document] = []

    def ingest(self, documents: list[Document]) -> IngestStats:
        """Records the documents and reports two chunks each.

        Args:
            documents: Documents to index.

        Returns:
            stats: Counts for the recorded run.
        """
        self.documents = documents
        return IngestStats(documents=len(documents), chunks=len(documents) * 2, seconds=0.1)


@pytest.fixture
def ingestor() -> RecordingIngestor:
    """Builds a fresh recording ingestor.

    Returns:
        ingestor: Double capturing ingest() calls.
    """
    return RecordingIngestor()


@pytest.fixture
def client(ingestor: RecordingIngestor) -> TestClient:
    """Builds a client whose uploads reach the recording ingestor.

    Args:
        ingestor: Double to install in place of the live ingestor.

    Returns:
        client: TestClient bound to the app.
    """
    app = create_app()
    app.dependency_overrides[build_ingestor] = lambda: ingestor
    return TestClient(app)


def test_indexes_an_uploaded_document(client: TestClient, ingestor: RecordingIngestor) -> None:
    """Verifies a text upload reaches the ingestor with its content intact."""
    files = [("files", ("clause.txt", b"termination for convenience", "text/plain"))]
    assert client.post("/documents", files=files).status_code == 200
    assert [d.title for d in ingestor.documents] == ["clause.txt"]
    assert ingestor.documents[0].text == "termination for convenience"


def test_reports_chunks_written(client: TestClient) -> None:
    """Verifies the response says what was indexed."""
    files = [("files", ("a.md", b"# heading", "text/markdown"))]
    body = client.post("/documents", files=files).json()
    assert body["documents"][0]["doc_id"] == "a"
    assert body["chunks"] == 2
    assert body["skipped"] == {}


def test_skips_an_unsupported_type(client: TestClient, ingestor: RecordingIngestor) -> None:
    """Verifies a type the pipeline cannot chunk is reported rather than indexed."""
    files = [
        ("files", ("notes.txt", b"keep me", "text/plain")),
        ("files", ("scan.pdf", b"%PDF-1.7", "application/pdf")),
    ]
    body = client.post("/documents", files=files).json()
    assert [d.title for d in ingestor.documents] == ["notes.txt"]
    assert "scan.pdf" in body["skipped"]


def test_skips_a_file_that_is_not_utf8(client: TestClient) -> None:
    """Verifies undecodable bytes are reported instead of raising."""
    files = [
        ("files", ("ok.txt", b"fine", "text/plain")),
        ("files", ("bad.txt", b"\xff\xfe\x00\x01", "text/plain")),
    ]
    body = client.post("/documents", files=files).json()
    assert "bad.txt" in body["skipped"]


def test_one_bad_file_does_not_lose_the_batch(
    client: TestClient, ingestor: RecordingIngestor
) -> None:
    """Verifies a rejected file does not stop the rest of the upload."""
    files = [
        ("files", ("good.txt", b"indexed", "text/plain")),
        ("files", ("bad.bin", b"\x00\x01", "application/octet-stream")),
    ]
    assert client.post("/documents", files=files).status_code == 200
    assert [d.title for d in ingestor.documents] == ["good.txt"]


def test_rejects_an_upload_with_nothing_indexable(client: TestClient) -> None:
    """Verifies an upload of only unsupported files answers 400."""
    files = [("files", ("scan.pdf", b"%PDF-1.7", "application/pdf"))]
    assert client.post("/documents", files=files).status_code == 400
