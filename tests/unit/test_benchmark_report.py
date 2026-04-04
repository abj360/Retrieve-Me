#!/usr/bin/env python3
"""
test_benchmark_report.py --- unit tests for the benchmark table generator

Contains:
    test_render_table_has_one_row_per_run(): asserts table shape
    test_load_results_parses_fields(): asserts JSON fields map onto the dataclass
"""

from scripts.benchmark_report import BenchmarkResult, render_table

RUN = BenchmarkResult(
    name="hybrid",
    dataset="legal-500",
    ndcg_at_10=0.63,
    recall_at_50=0.79,
    p50_ms=71,
    p95_ms=132,
    p99_ms=190,
    delta_ndcg=0.22,
)


def test_render_table_has_one_row_per_run() -> None:
    """Asserts the table renders a header, divider, and one row per run."""
    lines = render_table([RUN]).splitlines()
    assert len(lines) == 3
    assert "hybrid" in lines[2]


def test_load_results_parses_fields(tmp_path) -> None:
    """Asserts JSON fields map onto the dataclass."""
    from scripts.benchmark_report import load_results

    target = tmp_path / "results.json"
    target.write_text(
        '[{"name": "x", "dataset": "d", "ndcgAt10": 0.5, "recallAt50": 0.6,'
        ' "p50Ms": 10, "p95Ms": 20, "p99Ms": 30, "deltaNdcgVsBaseline": 0.1}]',
        encoding="utf-8",
    )
    (result,) = load_results(target)
    assert result.name == "x"
    assert result.delta_ndcg == 0.1
