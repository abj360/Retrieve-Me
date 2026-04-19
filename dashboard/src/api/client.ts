#!/usr/bin/env ts-node
/**
 * client.ts --- thin API client for the Retrieve-Me backend
 *
 * Contains:
 *   ApiError: error carrying the HTTP status of a failed request
 *   apiFetch: fetch wrapper with JSON handling and error propagation
 *   getBenchmarkRuns: loads benchmark runs from the static export
 */

import type { BenchmarkRun } from "../types";

const API_BASE = "/api";

/**
 * Error carrying the HTTP status of a failed API request.
 */
export class ApiError extends Error {
  status: number;

  constructor(path: string, status: number) {
    super(`request to ${path} failed with ${status}`);
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
  const response = await fetch(path, init);
  if (!response.ok) {
    throw new ApiError(path, response.status);
  }
  return (await response.json()) as T;
}

/**
 * Loads benchmark runs from the static export shipped with the dashboard.
 *
 * @returns runs - Recorded benchmark runs, newest last.
 */
export function getBenchmarkRuns(): Promise<BenchmarkRun[]> {
  return apiFetch<BenchmarkRun[]>("/benchmarks.json");
}

export { API_BASE };
