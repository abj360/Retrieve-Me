#!/usr/bin/env python3
"""
benchmark_report.py --- generates a markdown benchmark table from run results

Contains:
    BenchmarkResult: one benchmark run's metrics
    load_results(): reads benchmark results from a JSON file
    render_table(): renders results as a markdown table
    main(): CLI entrypoint
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    """Carries one benchmark run's metrics.

    Attributes:
        name: Run identifier.
        dataset: Dataset the run evaluated on.
        ndcg_at_10: nDCG at rank 10.
        recall_at_50: Recall at rank 50.
        p50_ms: Median end-to-end latency in milliseconds.
        p95_ms: 95th-percentile latency in milliseconds.
    """

    name: str
    dataset: str
    ndcg_at_10: float
    recall_at_50: float
    p50_ms: float
    p95_ms: float


def load_results(path: Path) -> list[BenchmarkResult]:
    """Reads benchmark results from a JSON file.

    Args:
        path: JSON file holding a list of run objects.

    Returns:
        results: Parsed benchmark results in file order.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        BenchmarkResult(
            name=run["name"],
            dataset=run["dataset"],
            ndcg_at_10=run["ndcgAt10"],
            recall_at_50=run["recallAt50"],
            p50_ms=run["p50Ms"],
            p95_ms=run["p95Ms"],
        )
        for run in raw
    ]


def render_table(results: list[BenchmarkResult]) -> str:
    """Renders results as a markdown table.

    Args:
        results: Benchmark results to tabulate.

    Returns:
        table: Markdown table with one row per run.
    """
    header = "| run | dataset | nDCG@10 | recall@50 | p50 (ms) | p95 (ms) |"
    divider = "|---|---|---|---|---|---|"
    rows = [
        f"| {run.name} | {run.dataset} | {run.ndcg_at_10:.2f} | "
        f"{run.recall_at_50:.2f} | {run.p50_ms:.0f} | {run.p95_ms:.0f} |"
        for run in results
    ]
    return "\n".join([header, divider, *rows])


def main() -> None:
    """Renders the benchmark table for a results JSON file."""
    parser = argparse.ArgumentParser(description="Render a benchmark table")
    parser.add_argument("--results", required=True, type=Path, help="results JSON file")
    args = parser.parse_args()
    print(render_table(load_results(args.results)))


if __name__ == "__main__":
    main()
