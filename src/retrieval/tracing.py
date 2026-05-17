#!/usr/bin/env python3
"""
tracing.py --- lightweight per-stage latency tracing for the pipeline

Contains:
    Span: one timed pipeline stage
    LatencyTracer: records spans and summarizes stage latency
    percentile(): linear-interpolation percentile helper
    LatencyTracer.report(): renders a per-stage latency table
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    """Carries one timed pipeline stage.

    Attributes:
        name: Stage name (embed, sparse, dense, fuse, rerank).
        duration_ms: Wall-clock duration of the stage in milliseconds.
    """

    name: str
    duration_ms: float


@dataclass
class LatencyTracer:
    """Records spans and summarizes stage latency.

    Attributes:
        spans: Timed spans recorded so far.
    """

    spans: list[Span] = field(default_factory=list)  # appended per traced stage

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Times one pipeline stage and records it as a span.

        Args:
            name: Stage name to record.
        """
        started = time.perf_counter()
        yield  # span closes even if the body raises
        self.spans.append(Span(name=name, duration_ms=(time.perf_counter() - started) * 1000))

    def summary(self) -> dict[str, float]:
        """Summarizes total milliseconds per span name.

        Returns:
            totals: Total duration per stage name.
        """
        totals: dict[str, float] = {}
        for span in self.spans:
            totals[span.name] = totals.get(span.name, 0.0) + span.duration_ms
        return totals


def percentile(values: list[float], p: float) -> float:
    """Computes the p-th percentile of a list of numbers.

    Args:
        values: Measurements to summarize.
        p: Percentile in [0, 100].

    Returns:
        percentile: Value at the p-th percentile (linear interpolation).
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


    def report(self) -> str:
        """Renders a per-stage latency report as text.

        Returns:
            report: Per-stage totals plus p50/p95 per span name.
        """
        by_name: dict[str, list[float]] = {}
        for span in self.spans:
            by_name.setdefault(span.name, []).append(span.duration_ms)
        lines = ["stage | count | p50 (ms) | p95 (ms)", "------+-------+----------+----------"]
        for name, durations in sorted(by_name.items()):
            lines.append(
                f"{name} | {len(durations)} | {percentile(durations, 50):.1f} "
                f"| {percentile(durations, 95):.1f}"
            )
        return "\n".join(lines)  # one row per stage, sorted by name
