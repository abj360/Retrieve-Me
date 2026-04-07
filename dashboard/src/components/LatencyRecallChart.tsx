#!/usr/bin/env ts-node
/**
 * LatencyRecallChart.tsx --- latency and recall comparison across benchmark runs
 *
 * Contains:
 *   LatencyRecallChart: plots p95 latency against nDCG for each run
 */

import {
  CartesianGrid,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
} from "recharts";

import type { BenchmarkRun } from "../types";

interface LatencyRecallChartProps {
  runs: BenchmarkRun[];
}

/**
 * Plots p95 latency bars and nDCG line for each benchmark run.
 *
 * @param props - Benchmark runs to plot.
 * @returns element - Chart element.
 */
export function LatencyRecallChart({ runs }: LatencyRecallChartProps) {
  const latest = runs[runs.length - 1];
  const points = runs.map((run) => ({
    name: run.name,
    p95Ms: run.p95Ms,
    ndcgAt10: run.ndcgAt10,
  }));

  return (
    <section className="panel chart-container">
      <h2>Latency vs recall</h2>
      <p className="chart-subtitle">Latest run: {latest.name}</p>
      <ComposedChart width={720} height={320} data={points}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis yAxisId="left" />
        <YAxis yAxisId="right" orientation="right" />
        <Bar yAxisId="left" dataKey="p95Ms" fill="var(--chart-bar)" />
        <Line yAxisId="right" dataKey="ndcgAt10" stroke="var(--chart-line)" />
      </ComposedChart>
    </section>
  );
}
