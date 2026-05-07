#!/usr/bin/env python3
"""
test_middleware.py --- unit tests for the request logging middleware

Contains:
    test_request_id_generated_when_missing(): asserts a request id is minted
    test_request_id_echoed_when_provided(): asserts a provided id round-trips
"""

from fastapi.testclient import TestClient

from src.api.main import create_app


def test_request_id_generated_when_missing() -> None:
    """Asserts responses carry a freshly minted X-Request-ID."""
    response = TestClient(create_app()).get("/healthz")
    assert response.headers["X-Request-ID"]


def test_request_id_echoed_when_provided() -> None:
    """Asserts a caller-provided X-Request-ID is returned unchanged."""
    response = TestClient(create_app()).get("/healthz", headers={"X-Request-ID": "rid-123"})
    assert response.headers["X-Request-ID"] == "rid-123"


def test_health_probe_not_logged_at_info(caplog) -> None:
    """Asserts health probes skip request logging once quiet paths land."""
    import logging

    with caplog.at_level(logging.INFO, logger="retrieval.http"):
        TestClient(create_app()).get("/healthz")
    assert not [record for record in caplog.records if "/healthz" in record.getMessage()]


def test_generated_request_id_is_twelve_chars() -> None:
    """Asserts minted request ids are twelve hex characters."""
    response = TestClient(create_app()).get("/healthz")
    assert len(response.headers["X-Request-ID"]) == 12
