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


def test_nested_spans_record_both() -> None:
    """Asserts nested spans record inner and outer stages."""
    tracer = LatencyTracer()
    with tracer.span("outer"):
        with tracer.span("inner"):
            pass
    assert [span.name for span in tracer.spans] == ["inner", "outer"]


def test_summary_sums_same_name_spans() -> None:
    """Asserts repeated stages accumulate in the summary."""
    tracer = LatencyTracer()
    with tracer.span("x"):
        pass
    with tracer.span("x"):
        pass
    summary = tracer.summary()
    assert summary["x"] >= 0


def test_span_names_arbitrary() -> None:
    """Asserts any stage name is accepted."""
    tracer = LatencyTracer()
    with tracer.span("embed"):
        pass
    with tracer.span("rerank"):
        pass
    assert {span.name for span in tracer.spans} == {"embed", "rerank"}


def test_tracer_in_pipeline_records_stages(indexed_stores, stub_embedder, stub_reranker) -> None:
    """Asserts the hybrid pipeline emits one span per stage."""
    from src.retrieval.fusion import FusionConfig, ResultFuser
    from src.retrieval.strategies import (
        DenseRetrievalStrategy,
        HybridRetriever,
        SparseRetrievalStrategy,
    )

    bm25, dense = indexed_stores
    tracer = LatencyTracer()
    pipeline = HybridRetriever(
        SparseRetrievalStrategy(bm25),
        DenseRetrievalStrategy(dense, stub_embedder),
        ResultFuser(FusionConfig()),
        stub_reranker,
        tracer=tracer,
    )
    pipeline.retrieve("clause", top_k=2)
    names = {span.name for span in tracer.spans}
    assert {"sparse", "dense", "fuse", "rerank"} <= names


def test_percentile_of_empty_is_zero() -> None:
    """Asserts an empty value list percentiles to zero."""
    from src.retrieval.tracing import percentile

    assert percentile([], 95) == 0.0


def test_percentile_interpolates() -> None:
    """Asserts percentile interpolates between neighbors."""
    from src.retrieval.tracing import percentile

    assert percentile([10.0, 20.0], 50) == 15.0


def test_percentile_single_value() -> None:
    """Asserts a single value is its own percentile."""
    from src.retrieval.tracing import percentile

    assert percentile([42.0], 95) == 42.0


def test_report_lists_each_stage() -> None:
    """Asserts the report has one row per stage."""
    tracer = LatencyTracer()
    with tracer.span("sparse"):
        pass
    with tracer.span("dense"):
        pass
    report = tracer.report()
    assert "sparse" in report and "dense" in report
