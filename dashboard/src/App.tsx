#!/usr/bin/env ts-node
/**
 * App.tsx --- root component of the Retrieve-Me benchmark dashboard
 *
 * Contains:
 *   App: renders the header and the benchmark results table
 */

import { useState } from "react";

import type { BenchmarkRun } from "./types";

const BENCHMARK_RUNS: BenchmarkRun[] = [
  {
    id: "run-001",
    name: "dense-only-baseline",
    dataset: "legal-500",
    ndcgAt10: 0.41,
    recallAt50: 0.58,
    p50Ms: 32,
    p95Ms: 61,
    deltaNdcgVsBaseline: 0,
    ranAt: "2026-03-29T09:14:00Z",
  },
  {
    id: "run-002",
    name: "hybrid-rrf-baseline",
    dataset: "legal-500",
    ndcgAt10: 0.52,
    recallAt50: 0.71,
    p50Ms: 48,
    p95Ms: 96,
    deltaNdcgVsBaseline: 0.11,
    ranAt: "2026-03-29T11:02:00Z",
  },
];

const PAGE_SIZE = 10;

/**
 * Renders the dashboard header and the benchmark results table.
 *
 * @returns element - Root application element.
 */
export function App() {
  const [page, setPage] = useState(0);
  const totalPages = Math.floor(BENCHMARK_RUNS.length / PAGE_SIZE);
  const visibleRuns = BENCHMARK_RUNS.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Retrieve-Me</h1>
        <p>Hybrid retrieval benchmarks: BM25 + dense + cross-encoder rerank</p>
      </header>
      <main>
        <section className="panel">
          <h2>Benchmark runs</h2>
          <table className="results-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Dataset</th>
                <th>nDCG@10</th>
                <th>Recall@50</th>
                <th>p95 (ms)</th>
                <th>Δ nDCG</th>
              </tr>
            </thead>
            <tbody>
              {visibleRuns.map((run) => (
                <tr key={run.id}>
                  <td>{run.name}</td>
                  <td>{run.dataset}</td>
                  <td>{run.ndcgAt10.toFixed(2)}</td>
                  <td>{run.recallAt50.toFixed(2)}</td>
                  <td>{run.p95Ms}</td>
                  <td>{run.deltaNdcgVsBaseline > 0 ? `+${run.deltaNdcgVsBaseline.toFixed(2)}` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="pagination">
            <button onClick={() => setPage(page - 1)} disabled={page === 0}>
              Previous
            </button>
            <span>
              Page {page + 1} of {totalPages}
            </span>
            <button onClick={() => setPage(page + 1)} disabled={page + 1 >= totalPages}>
              Next
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}
