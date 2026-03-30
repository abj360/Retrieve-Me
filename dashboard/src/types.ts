#!/usr/bin/env ts-node
/**
 * types.ts --- shared types for the benchmark dashboard
 *
 * Contains:
 *   BenchmarkRun: one recorded benchmark run with retrieval metrics
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
