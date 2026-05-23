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
    test_filters_are_echoed(): asserts applied filters round-trip in the response
    test_healthz_reports_liveness(): asserts /healthz returns 200 with version info
    test_readyz_shape(): asserts /readyz reports per-dependency states
    test_validation_error_body(): asserts 422 bodies carry a detail field
    test_get_on_retrieve_is_405(): asserts the endpoint is POST-only
    test_pipeline_failure_returns_502(): asserts backend failures surface as 502
    test_request_id_header_present(): asserts X-Request-ID is set on responses
    test_cache_header_miss_then_hit(): asserts X-Cache flips from MISS to HIT
    test_response_is_json(): asserts the content type is JSON
"""

from fastapi.testclient import TestClient

from src.api.cache import InMemoryQueryCache
from src.api.dependencies import get_pipeline, get_query_cache
from src.api.main import create_app
from src.retrieval.fusion import RankedResult

FAKE_RESULT_COUNT = 30
FAKE_SCORE_STEP = 0.01


class FakePipeline:
    """Returns deterministic ranked results for contract tests."""

    def retrieve(self, query: str, top_k: int = 10, filters: dict | None = None) -> list:
        """Returns thirty deterministic results for any query.

        Args:
            query: Query text echoed into the fake passages.
            top_k: Maximum number of chunks to return.
            filters: Ignored by the fake.

        Returns:
            results: Deterministic ranked results with decreasing scores.
        """
        return [
            RankedResult(
                chunk_id=f"chunk-{index:03d}",
                doc_id=f"doc-{index}",
                text=f"passage {index} for {query}",
                score=1.0 - index * FAKE_SCORE_STEP,
                source="fused",
            )
            for index in range(FAKE_RESULT_COUNT)
        ]


def client() -> TestClient:
    """Builds a TestClient with pipeline and cache overridden.

    Returns:
        test_client: Client bound to an app with fake dependencies.
    """
    app = create_app()
    app.dependency_overrides[get_pipeline] = FakePipeline()
    app.dependency_overrides[get_query_cache] = InMemoryQueryCache()
    return TestClient(app)


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


def test_filters_are_echoed() -> None:
    """Asserts applied_filters round-trips the request filters."""
    filters = {"source": "legal"}
    response = client().post("/retrieve", json={"query": "clause", "filters": filters})
    assert response.status_code == 200
    assert response.json()["applied_filters"] == filters


def test_healthz_reports_liveness() -> None:
    """Asserts /healthz returns 200 with a status payload."""
    response = client().get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_shape() -> None:
    """Asserts /readyz reports per-dependency states or fails closed."""
    response = client().get("/readyz")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert set(response.json()) >= {"status", "qdrant", "redis"}
    else:
        assert "failed" in response.json()["detail"]


def test_validation_error_body() -> None:
    """Asserts validation failures return a detail array."""
    response = client().post("/retrieve", json={"query": "x", "page": 0})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_get_on_retrieve_is_405() -> None:
    """Asserts GET on /retrieve is rejected as method-not-allowed."""
    response = client().get("/retrieve")
    assert response.status_code == 405


def test_pipeline_failure_returns_502() -> None:
    """Asserts a failing pipeline surfaces as 502, not 500."""
    app = create_app()

    class FailingPipeline:
        """Raises on every retrieval call."""

        def retrieve(self, query: str, top_k: int = 10, filters: dict | None = None) -> list:
            """Raises a backend failure for the contract test."""
            raise RuntimeError("backend down")

    app.dependency_overrides[get_pipeline] = FailingPipeline()
    app.dependency_overrides[get_query_cache] = InMemoryQueryCache()
    response = TestClient(app).post("/retrieve", json={"query": "clause"})
    assert response.status_code == 502
    assert response.json()["detail"] == "retrieval backend unavailable"


def test_request_id_header_present() -> None:
    """Asserts responses carry an X-Request-ID header."""
    response = client().post("/retrieve", json={"query": "clause"})
    assert "X-Request-ID" in response.headers


def test_cache_header_miss_then_hit() -> None:
    """Asserts X-Cache is MISS on first call and HIT on repeat."""
    first = client().post("/retrieve", json={"query": "cache check"})
    second = client().post("/retrieve", json={"query": "cache check"})
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"


def test_response_is_json() -> None:
    """Asserts the response content type is application/json."""
    response = client().post("/retrieve", json={"query": "clause"})
    assert response.headers["content-type"].startswith("application/json")


def test_retrieve_rejects_empty_query() -> None:
    """Asserts a missing query field fails validation with 422."""
    response = client().post("/retrieve", json={})
    assert response.status_code == 422


def test_top_k_larger_than_matches_returns_all() -> None:
    """Asserts a top_k above the match count returns every match."""
    response = client().post("/retrieve", json={"query": "clause", "top_k": 100})
    assert response.status_code == 200
    assert response.json()["total"] == 30


def test_results_have_nonempty_text() -> None:
    """Asserts every returned chunk carries non-empty text."""
    response = client().post("/retrieve", json={"query": "clause"})
    assert all(chunk["text"] for chunk in response.json()["results"])


def test_took_ms_is_nonnegative() -> None:
    """Asserts took_ms is a non-negative number."""
    response = client().post("/retrieve", json={"query": "clause"})
    assert response.json()["took_ms"] >= 0


def test_filters_default_to_null() -> None:
    """Asserts applied_filters is null when no filters are sent."""
    response = client().post("/retrieve", json={"query": "clause"})
    assert response.json()["applied_filters"] is None


def test_top_k_one_returns_single_result() -> None:
    """Asserts top_k of 1 yields exactly one result."""
    response = client().post("/retrieve", json={"query": "clause", "top_k": 1})
    assert len(response.json()["results"]) == 1


def test_page_size_one_paginates() -> None:
    """Asserts page_size of 1 paginates one result per page."""
    page_one = client().post("/retrieve", json={"query": "clause", "page_size": 1})
    page_two = client().post("/retrieve", json={"query": "clause", "page_size": 1, "page": 2})
    assert len(page_one.json()["results"]) == 1
    assert page_one.json()["results"][0]["chunk_id"] != page_two.json()["results"][0]["chunk_id"]


def test_results_carry_source_field() -> None:
    """Asserts every chunk reports which retrieval stage produced it."""
    response = client().post("/retrieve", json={"query": "clause"})
    assert all(chunk["source"] for chunk in response.json()["results"])


def test_has_next_flags_more_pages() -> None:
    """Asserts has_next is true on page 1 and false on the last page."""
    first = client().post("/retrieve", json={"query": "clause", "page_size": 10})
    last = client().post("/retrieve", json={"query": "clause", "page": 3, "page_size": 10})
    assert first.json()["has_next"] is True
    assert last.json()["has_next"] is False


def test_page_beyond_results_returns_empty_list() -> None:
    """Asserts an out-of-range page yields an empty result list."""
    response = client().post("/retrieve", json={"query": "clause", "page": 99})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_empty_string_query_rejected() -> None:
    """Asserts a blank query string fails validation."""
    response = client().post("/retrieve", json={"query": ""})
    assert response.status_code == 422


def test_missing_body_rejected() -> None:
    """Asserts a request without a body fails validation."""
    response = client().post("/retrieve")
    assert response.status_code == 422


def test_502_detail_is_string() -> None:
    """Asserts the 502 error body carries a string detail."""
    app = create_app()

    class FailingPipeline:
        """Raises on every retrieval call."""

        def retrieve(self, query: str, top_k: int = 10, filters: dict | None = None) -> list:
            """Raises a backend failure for the contract test."""
            raise RuntimeError("backend down")

    app.dependency_overrides[get_pipeline] = FailingPipeline()
    app.dependency_overrides[get_query_cache] = InMemoryQueryCache()
    response = TestClient(app).post("/retrieve", json={"query": "clause"})
    assert isinstance(response.json()["detail"], str)


def test_has_next_false_on_exact_multiple() -> None:
    """Asserts has_next is false when the last page is exactly full."""
    response = client().post("/retrieve", json={"query": "clause", "page": 3, "page_size": 10})
    assert response.json()["has_next"] is False


def test_page_size_upper_bound_accepted() -> None:
    """Asserts page_size of 100 passes validation."""
    response = client().post("/retrieve", json={"query": "clause", "page_size": 100})
    assert response.status_code == 200


def test_healthz_includes_version() -> None:
    """Asserts /healthz reports the service version."""
    response = client().get("/healthz")
    assert response.status_code == 200
    assert "version" in response.json()
