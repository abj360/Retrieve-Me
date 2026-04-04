#!/usr/bin/env python3
"""
test_tracing.py --- unit tests for the latency tracer

Contains:
    test_span_records_duration(): asserts spans carry a duration
    test_summary_totals_by_name(): asserts summary groups by span name
    test_empty_summary(): asserts an empty tracer summarizes to {}
"""

from src.retrieval.tracing import LatencyTracer


def test_span_records_duration() -> None:
    """Asserts spans carry a non-negative duration."""
    tracer = LatencyTracer()
    with tracer.span("stage"):
        pass
    assert tracer.spans[0].name == "stage"
    assert tracer.spans[0].duration_ms >= 0


def test_summary_totals_by_name() -> None:
    """Asserts summary groups durations by span name."""
    tracer = LatencyTracer()
    with tracer.span("a"):
        pass
    with tracer.span("a"):
        pass
    with tracer.span("b"):
        pass
    summary = tracer.summary()
    assert set(summary) == {"a", "b"}


def test_empty_summary() -> None:
    """Asserts an empty tracer summarizes to {}."""
    assert LatencyTracer().summary() == {}
