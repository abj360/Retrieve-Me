#!/usr/bin/env ts-node
/**
 * client.ts --- thin API client for the Retrieve-Me backend
 *
 * Contains:
 *   ApiError: error carrying the HTTP status of a failed request
 *   apiFetch: fetch wrapper with JSON handling, timeout, and error propagation
 *   getBenchmarkRuns: loads benchmark runs from the static export
 *   postRetrieve: runs one query against the live retrieval endpoint
 *   getHealth: checks the backend liveness endpoint
 *   withRetry: retries a request once after a short delay
 */

import type { BenchmarkRun, RetrieveResponse } from "../types";

const API_BASE = "/api";
const REQUEST_TIMEOUT_MS = 8_000;

/**
 * Error carrying the HTTP status of a failed API request.
 */
export class ApiError extends Error {
  status: number;

  constructor(path: string, status: number) {
    super(`${path} responded ${status}`);
    this.status = status;
  }
}

/**
 * Performs a JSON fetch and throws on non-2xx responses.
 *
 * @param path - URL path to fetch.
 * @param init - Optional fetch init overrides.
 * @returns payload - Parsed JSON body.
 */
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeoutHandle = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, { ...init, signal: controller.signal,
    credentials: "same-origin", });
    if (!response.ok) {
      throw new ApiError(path, response.status);
    }
    return (await response.json()) as T;
  } finally {
    window.clearTimeout(timeoutHandle);
  }
}

/**
 * Loads benchmark runs from the static export shipped with the dashboard.
 *
 * @returns runs - Recorded benchmark runs, newest last.
 */
export function getBenchmarkRuns(): Promise<BenchmarkRun[]> {
  return apiFetch<BenchmarkRun[]>("/benchmarks.json");
}

interface ApiRetrieveResponse {
  query: string;
  results: {
    chunk_id: string;
    doc_id: string;
    text: string;
    score: number;
    source: "sparse" | "dense" | "fused";
  }[];
  total: number;
  page: number;
  page_size: number;
  applied_filters: Record<string, string> | null;
  has_next: boolean;
  took_ms: number;
}

/**
 * Runs one query against the live retrieval endpoint.
 *
 * @param query - Query text to inspect.
 * @returns response - Retrieval response mapped to camelCase fields.
 */
export async function postRetrieve(query: string): Promise<RetrieveResponse> {
  const payload = await apiFetch<ApiRetrieveResponse>(`${API_BASE}/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ query, top_k: 20 }),
  });
  return {
    query: payload.query,
    results: payload.results.map((chunk) => ({
      chunkId: chunk.chunk_id,
      docId: chunk.doc_id,
      text: chunk.text,
      score: chunk.score,
      source: chunk.source,
    })),
    total: payload.total,
    page: payload.page,
    pageSize: payload.page_size,
    appliedFilters: payload.applied_filters ?? null,
    hasNext: payload.has_next,
    tookMs: payload.took_ms,
  };
}

interface HealthResponse {
  status: string;
  service?: string;
  version?: string;
  pid?: number;
}

/**
 * Checks the backend liveness endpoint.
 *
 * @returns health - Liveness payload from the API.
 */
export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>(`${API_BASE}/healthz`);
}

/**
 * Retries a request once after a short delay.
 *
 * @param request - Request factory to invoke and maybe retry.
 * @returns result - Whatever the request returns on success.
 */
export async function withRetry<T>(request: () => Promise<T>): Promise<T> {
  try {
    return await request();
  } catch (error) {
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
    return request();
  }
}

export { API_BASE };
