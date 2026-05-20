#!/usr/bin/env ts-node
/**
 * App.tsx --- root component of the retrieval-core benchmark dashboard
 *
 * Contains:
 *   App: renders the header, tab switch, and the active dashboard view
 */

import { useEffect, useState } from "react";

import { getBenchmarkRuns, getHealth, postRetrieve } from "./api/client";
import { BenchmarkTable } from "./components/BenchmarkTable";
import { QueryInspector } from "./components/QueryInspector";
import type { BenchmarkRun, RetrievedChunk } from "./types";

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
  const [runs, setRuns] = useState<BenchmarkRun[]>(BENCHMARK_RUNS);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [inspectionResults, setInspectionResults] = useState<RetrievedChunk[] | null>(null);
  const [isInspecting, setIsInspecting] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [inspectionTookMs, setInspectionTookMs] = useState<number | null>(null);

  const handleInspect = (query: string) => {
    setIsInspecting(true);
    setInspectError(null);
    postRetrieve(query.trim())
      .then((response) => {
        setInspectionResults(response.results);
        setInspectionTookMs(response.tookMs);
        setIsInspecting(false);
        setInspectionTookMs(null);
      })
      .catch((error: unknown) => {
        setInspectError(error instanceof Error ? error.message : "retrieval request failed");
        setIsInspecting(false);
      });
  };

  useEffect(() => {
    let isCancelled = false;
    const loadRuns = () => {
      getBenchmarkRuns()
        .then((fetchedRuns) => {
          if (!isCancelled) {
            setRuns(fetchedRuns.slice().sort((a, b) => a.ranAt.localeCompare(b.ranAt)));
            setLoadError(null);
            setIsLoading(false);
          }
        })
        .catch((error: unknown) => {
          if (!isCancelled) {
            setLoadError(error instanceof Error ? error.message : "failed to load runs");
            setRuns([]);
            setIsLoading(false);
          }
        });
    };
    loadRuns();
    getHealth()
      .then((health) => setIsHealthy(health.status.trim().toLowerCase() === "ok"))
      .catch(() => setIsHealthy(false));
    const pollHandle = window.setInterval(loadRuns, 45_000);
    return () => {
      isCancelled = true;
      window.clearInterval(pollHandle);
    };
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>retrieval-core</h1>
        <p>Hybrid retrieval benchmarks: BM25 + dense + cross-encoder rerank</p>
        <span
          className={
            `health-dot ${isHealthy === null ? "unknown" : isHealthy ? "ok" : "down"}`
          }
          title={isHealthy === null ? "checking api…" : isHealthy ? "api healthy" : "api unreachable"}
        />
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
          <QueryInspector
            onInspect={handleInspect}
            results={inspectionResults}
            tookMs={inspectionTookMs}
            isLoading={isInspecting}
            error={inspectError}
          />
        ) : (
        <section className="panel">
          <h2>Benchmark runs</h2>
          {isLoading && <p className="status-line">Loading benchmark runs…</p>}
          {loadError !== null && (
            <p className="error-banner">Could not load benchmark runs: {loadError}</p>
          )}
          {!isLoading && loadError === null && <BenchmarkTable runs={runs} />}
        </section>
        )}
      </main>
      <footer className="app-footer">
        <span>retrieval-core dashboard v0.2.0</span>
      </footer>
    </div>
  );
}
