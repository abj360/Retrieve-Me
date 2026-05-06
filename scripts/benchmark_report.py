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
        p99_ms: 99th-percentile latency in milliseconds.
        delta_ndcg: nDCG delta against the dense-only baseline.
    """

    name: str
    dataset: str
    ndcg_at_10: float
    recall_at_50: float
    p50_ms: float
    p95_ms: float
    p99_ms: float = 0.0  # optional: older runs lack p99
    delta_ndcg: float = 0.0


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
            p99_ms=run.get("p99Ms", 0.0),
            delta_ndcg=run.get("deltaNdcgVsBaseline", 0.0),
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
    header = "| run | dataset | nDCG@10 | recall@50 | p50 (ms) | p95 (ms) | p99 (ms) | Δ nDCG |"
    divider = "|---|---|---|---|---|---|---|---|"
    rows = [
        f"| {run.name} | {run.dataset} | {run.ndcg_at_10:.2f} | "
        f"{run.recall_at_50:.2f} | {run.p50_ms:.0f} | {run.p95_ms:.0f} | "
        f"{run.p99_ms:.0f} | {run.delta_ndcg:+.2f} |"
        for run in results
    ]
    return "\n".join([header, divider, *rows])


def main() -> None:
    """Renders the benchmark table for a results JSON file."""
    parser = argparse.ArgumentParser(description="Render a benchmark table")
    parser.add_argument("--results", required=True, type=Path, help="results JSON file")
    parser.add_argument("--sort", choices=["name", "ndcg"], default="name")
    parser.add_argument(
        "--emit-dashboard-json",
        type=Path,
        default=None,
        help="also write the runs JSON the dashboard serves",
    )
    args = parser.parse_args()
    results = load_results(args.results)
    if args.sort == "ndcg":
        results = sorted(results, key=lambda run: run.ndcg_at_10, reverse=True)
    else:
        results = sorted(results, key=lambda run: run.name)
    print(render_table(results))
    if args.emit_dashboard_json is not None:
        payload = [
            {
                "id": f"run-{index:03d}",
                "name": run.name,
                "dataset": run.dataset,
                "ndcgAt10": run.ndcg_at_10,
                "recallAt50": run.recall_at_50,
                "p50Ms": run.p50_ms,
                "p95Ms": run.p95_ms,
                "deltaNdcgVsBaseline": run.delta_ndcg,
                "ranAt": "",
            }
            for index, run in enumerate(results, start=1)
        ]
        args.emit_dashboard_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
