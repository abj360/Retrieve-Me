#!/usr/bin/env ts-node
/**
 * QueryInspector.tsx --- inspection view for ad-hoc retrieval queries
 *
 * Contains:
 *   QueryInspector: renders a query box and live inspection results
 */

import { useState } from "react";

import type { RetrievedChunk, StageBreakdown } from "../types";

/**
 * Renders the query inspection view with mock results.
 *
 * @returns element - Query inspection panel.
 */
interface QueryInspectorProps {
  onInspect: (query: string) => void;
  results: RetrievedChunk[] | null;
  tookMs: number | null;
  isLoading: boolean;
  error: string | null;
}

/**
 * Renders the query inspection view driven by the live retrieval API.
 *
 * @param props - Inspection handler, results, and request state.
 * @returns element - Query inspection panel.
 */
export function QueryInspector({ onInspect, results, tookMs, isLoading, error }: QueryInspectorProps) {
  const [query, setQuery] = useState("");
  const visibleResults = results ?? [];
  const stageCounts = visibleResults.reduce<StageBreakdown>(
    (counts, chunk) => ({ ...counts, [chunk.source]: counts[chunk.source] + 1 }),
    { sparse: 0, dense: 0, fused: 0 },
  );

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onInspect(query);
  };

  return (
    <section className="panel query-inspector">
      <h2>Query inspector</h2>
      <form className="query-box" role="search" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Inspect a query, e.g. clause 3.1…"
          aria-label="query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && onInspect(query)}
        />
        <input type="text" placeholder="filters, e.g. source=legal" aria-label="metadata filters (optional)" className="filters-input" />
        <button type="submit" disabled={isLoading} aria-label="inspect query">
          {isLoading ? "Inspecting…" : "Inspect"}
        </button>
      </form>
      {tookMs !== null && <span className="timing-chip" title="end-to-end">{tookMs.toFixed(0)} ms</span>}
      {error !== null && <p className="error-banner">Retrieval failed: {error}</p>}
      <div className="stage-breakdown" aria-label="stage counts">
        {(["sparse", "dense", "fused"] as const).map((stage) => (
          <span key={stage} className={`badge stage-badge source-${stage}`}>
            {stage} · {stageCounts[stage] ?? 0}
          </span>
        ))}
      </div>
      {results !== null && visibleResults.length === 0 && !isLoading && error === null && (
        <p className="status-line">No chunks matched that query — try widening the filters.</p>
      )}
      <ol className="result-list" aria-label="inspection results">
        {visibleResults.map((chunk) => (
          <li key={chunk.chunkId} className="result-item">
            <span className={`badge source-${chunk.source}`}>{chunk.source}</span>
            <span className="result-text">{chunk.text}</span>
            <span
              className="score-bar"
              title={chunk.score.toFixed(2)}
              style={{ width: `${Math.round(chunk.score * 100)}%` }}
            />
            <span className="result-score">{chunk.score.toFixed(3)}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
