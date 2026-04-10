#!/usr/bin/env ts-node
/**
 * QueryInspector.tsx --- inspection view for ad-hoc retrieval queries
 *
 * Contains:
 *   QueryInspector: renders a query box and mock inspection results
 */

import { useState } from "react";

import type { RetrievedChunk } from "../types";

const MOCK_RESULTS: RetrievedChunk[] = [
  {
    chunkId: "license-agreement-chunk-0",
    docId: "license-agreement",
    text: "Section 3.1 The licensee shall indemnify the vendor against claims.",
    score: 0.92,
    source: "fused",
  },
  {
    chunkId: "rfc-7807-chunk-0",
    docId: "rfc-7807",
    text: "RFC 7807 defines problem details for HTTP APIs.",
    score: 0.74,
    source: "sparse",
  },
];

/**
 * Renders the query inspection view with mock results.
 *
 * @returns element - Query inspection panel.
 */
export function QueryInspector() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState<string | null>(null);
  const visibleResults =
    submitted === null
      ? MOCK_RESULTS
      : MOCK_RESULTS.filter((chunk) =>
          chunk.text.toLowerCase().includes(submitted.toLowerCase()),
        );

  return (
    <section className="panel query-inspector">
      <h2>Query inspector</h2>
      <div className="query-box">
        <input
          type="text"
          placeholder='Try a query, e.g. "indemnity clause"'
          aria-label="query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && setSubmitted(query)}
        />
        <button type="button" onClick={() => setSubmitted(query)}>
          Inspect
        </button>
      </div>
      <ol className="result-list" aria-label="inspection results">
        {visibleResults.map((chunk) => (
          <li key={chunk.chunkId} className="result-item">
            <span className={`badge source-${chunk.source}`}>{chunk.source}</span>
            <span className="result-text">{chunk.text}</span>
            <span className="result-score">{chunk.score.toFixed(2)}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
