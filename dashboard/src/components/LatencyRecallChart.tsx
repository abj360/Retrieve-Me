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
  Legend,
  Line,
  Bar,
  Tooltip,
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
    <section className="panel chart-container" aria-label="latency and recall chart">
      <h2>Latency vs recall (per run)</h2>
      <p className="chart-subtitle" title={latest.ranAt}>Latest: {latest.name} ({latest.dataset})</p>
      <ComposedChart width={760} height={320} data={points}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-20} />
        <YAxis yAxisId="left" label={{ value: "p95 (ms)", angle: -90, position: "insideLeft" }} />
        <YAxis
          yAxisId="right"
          orientation="right"
          domain={[0, 1]}
          label={{ value: "nDCG@10", angle: 90, position: "insideRight" }}
        />
        <Tooltip />
        <Legend />
        <Bar yAxisId="left" dataKey="p95Ms" name="p95 latency (ms)" fill="var(--chart-bar)" />
        <Line yAxisId="right" dataKey="ndcgAt10" name="nDCG@10" stroke="var(--chart-line)" />
      </ComposedChart>
    </section>
  );
}
