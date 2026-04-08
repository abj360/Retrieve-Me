#!/usr/bin/env ts-node
/**
 * App.tsx --- root component of the Retrieve-Me benchmark dashboard
 *
 * Contains:
 *   App: renders the header, tab switch, and the active dashboard view
 */

import { useState } from "react";

import { BenchmarkTable } from "./components/BenchmarkTable";
import { QueryInspector } from "./components/QueryInspector";
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

/**
 * Renders the dashboard header and the benchmark results table.
 *
 * @returns element - Root application element.
 */
type DashboardTab = "benchmarks" | "inspector";

export function App() {
  const [activeTab, setActiveTab] = useState<DashboardTab>("benchmarks");
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Retrieve-Me</h1>
        <p>Hybrid retrieval benchmarks: BM25 + dense + cross-encoder rerank</p>
      </header>
      <nav className="tabs">
        <button
          className={activeTab === "benchmarks" ? "tab active" : "tab"}
          onClick={() => setActiveTab("benchmarks")}
        >
          Benchmarks
        </button>
        <button
          className={activeTab === "inspector" ? "tab active" : "tab"}
          onClick={() => setActiveTab("inspector")}
        >
          Query inspector
        </button>
      </nav>
      <main>
        {activeTab === "inspector" ? (
          <QueryInspector />
        ) : (
        <section className="panel">
          <h2>Benchmark runs</h2>
          <BenchmarkTable runs={BENCHMARK_RUNS} />
        </section>
        )}
      </main>
    </div>
  );
}
