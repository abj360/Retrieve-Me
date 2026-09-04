/**
 * types.ts --- shared types for the benchmark dashboard
 *
 * Contains:
 *   BenchmarkRun: one recorded benchmark run with retrieval metrics
 *   RetrievedChunk: one scored chunk returned by the retrieval API
 *   RetrieveResponse: retrieval API response payload
 *   StageBreakdown: per-source result counts derived from a response
 *   UploadedDocument: one document accepted by an upload
 *   UploadResult: outcome of one document upload
 */

export interface BenchmarkRun {
  id: string;
  name: string;
  dataset: string;
  ndcgAt10: number;
  recallAt50: number;
  p50Ms: number;
  p95Ms: number;
  deltaNdcgVsBaseline: number;
  ranAt: string;
}

/**
 * RetrievedChunk: one scored chunk returned by the retrieval API.
 */
export interface RetrievedChunk {
  chunkId: string;
  docId: string;
  text: string;
  score: number;
  source: "sparse" | "dense" | "fused";
}

/**
 * RetrieveResponse: response payload of POST /retrieve.
 */
export interface RetrieveResponse {
  query: string;
  results: RetrievedChunk[];
  total: number;
  page: number;
  pageSize: number;
  appliedFilters: Record<string, string> | null;
  hasNext: boolean;
  tookMs: number;
}

/**
 * StageBreakdown: per-source result counts derived from a response.
 */
export interface StageBreakdown {
  sparse: number;
  dense: number;
  fused: number;
  total?: number;
}

/**
 * UploadedDocument: one document accepted by an upload.
 */
export interface UploadedDocument {
  docId: string;
  title: string;
  bytes: number;
}

/**
 * UploadResult: outcome of one document upload.
 */
export interface UploadResult {
  documents: UploadedDocument[];
  chunks: number;
  skipped: Record<string, string>;
  tookMs: number;
}
