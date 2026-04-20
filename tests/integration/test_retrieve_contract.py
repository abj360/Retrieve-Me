#!/usr/bin/env python3
"""
test_retrieve_contract.py --- contract tests for the retrieval endpoint

Contains:
    client(): provides a TestClient bound to the app
    test_retrieve_returns_expected_schema(): asserts response shape and types
    test_retrieve_respects_top_k(): asserts results are truncated to top_k
    test_retrieve_rejects_empty_query(): asserts missing queries fail validation
    test_pagination_pages_are_disjoint(): asserts page slices do not overlap
    test_pagination_reports_total(): asserts total reflects all matched chunks
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


def test_retrieve_default_top_k_returns_ten() -> None:
    """Asserts the default top_k of 10 yields ten results."""
    response = client().post("/retrieve", json={"query": "warranty"})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 10


def test_retrieve_scores_are_sorted_descending() -> None:
    """Asserts results arrive best-first by score."""
    response = client().post("/retrieve", json={"query": "liability"})
    scores = [chunk["score"] for chunk in response.json()["results"]]
    assert scores == sorted(scores, reverse=True)


def test_pagination_pages_are_disjoint() -> None:
    """Asserts page 1 and page 2 of the same query share no chunks."""
    page_one = client().post("/retrieve", json={"query": "clause", "page": 1, "page_size": 5})
    page_two = client().post("/retrieve", json={"query": "clause", "page": 2, "page_size": 5})
    ids_one = {chunk["chunk_id"] for chunk in page_one.json()["results"]}
    ids_two = {chunk["chunk_id"] for chunk in page_two.json()["results"]}
    assert ids_one.isdisjoint(ids_two)


def test_pagination_reports_total() -> None:
    """Asserts the total field counts every matched chunk."""
    response = client().post("/retrieve", json={"query": "clause", "page_size": 5})
    body = response.json()
    assert body["total"] == 30
    assert len(body["results"]) == 5


def test_retrieve_rejects_empty_query() -> None:
    """Asserts a missing query field fails validation with 422."""
    response = client().post("/retrieve", json={})
    assert response.status_code == 422
