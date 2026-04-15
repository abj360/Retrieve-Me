#!/usr/bin/env python3
"""
tracing.py --- lightweight per-stage latency tracing for the pipeline

Contains:
    Span: one timed pipeline stage
    LatencyTracer: records spans and summarizes stage latency
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

    spans: list[Span] = field(default_factory=list)

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
