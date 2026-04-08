#!/usr/bin/env ts-node
/**
 * BenchmarkTable.tsx --- benchmark runs table with retrieval metrics
 *
 * Contains:
 *   BenchmarkTable: renders benchmark runs through the generic ResultsTable
 */

import type { BenchmarkRun } from "../types";
import { ResultsTable } from "./ResultsTable";

interface BenchmarkTableProps {
  runs: BenchmarkRun[];
}

/**
 * Renders benchmark runs through the generic paginated table.
 *
 * @param props - Benchmark runs to display.
 * @returns element - Benchmark table element.
 */
export function BenchmarkTable({ runs }: BenchmarkTableProps) {
  return (
    <ResultsTable
      data={runs}
      rowKey={(run) => run.id}
      columns={[
        { label: "Run", render: (run) => run.name },
        { label: "Dataset", render: (run) => run.dataset },
        { label: "nDCG@10", render: (run) => run.ndcgAt10.toFixed(3) },
        { label: "Recall@50", render: (run) => run.recallAt50.toFixed(2) },
        { label: "p95 latency (ms)", render: (run) => run.p95Ms },
        {
          label: "Δ nDCG",
          render: (run) => (run.deltaNdcgVsBaseline > 0 ? `+${run.deltaNdcgVsBaseline.toFixed(2)}` : "—"),
        },
      ]}
    />
  );
}
