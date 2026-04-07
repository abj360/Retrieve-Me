#!/usr/bin/env python3
"""
test_retrieve_contract.py --- contract tests for the retrieval endpoint

Contains:
    client(): provides a TestClient bound to the app
    test_retrieve_returns_expected_schema(): asserts response shape and types
    test_retrieve_respects_top_k(): asserts results are truncated to top_k
    test_retrieve_rejects_empty_query(): asserts missing queries fail validation
"""

from fastapi.testclient import TestClient

from src.api.main import create_app


def client() -> TestClient:
    """Builds a TestClient for the service under test.

    Returns:
        test_client: Client bound to a fresh app instance.
    """
    return TestClient(create_app())


def test_retrieve_returns_expected_schema() -> None:
    """Asserts the response exposes the documented fields."""
    response = client().post("/retrieve", json={"query": "indemnity clause"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"query", "results", "took_ms"}
    first = body["results"][0]
    assert set(first) >= {"chunk_id", "doc_id", "text", "score"}


def test_retrieve_respects_top_k() -> None:
    """Asserts the endpoint returns at most top_k results."""
    response = client().post("/retrieve", json={"query": "termination", "top_k": 5})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 5


def test_retrieve_rejects_empty_query() -> None:
    """Asserts a missing query field fails validation with 422."""
    response = client().post("/retrieve", json={})
    assert response.status_code == 422
