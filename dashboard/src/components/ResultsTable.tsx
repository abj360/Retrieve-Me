#!/usr/bin/env ts-node
/**
 * ResultsTable.tsx --- generic paginated table for dashboard data
 *
 * Contains:
 *   ResultsTable: renders one page of rows with prev/next pagination
 */

import { useEffect, useState } from "react";

interface Column<T> {
  label: string;
  render: (item: T) => React.ReactNode;
}

interface ResultsTableProps<T> {
  data: T[];
  columns: Column<T>[];
  rowKey: (item: T) => string;
  pageSize?: number;
}

/**
 * Renders one page of rows with prev/next pagination.
 *
 * @param props - Rows, column definitions, key extractor, and page size.
 * @returns element - Paginated table element.
 */
export function ResultsTable<T>({ data, columns, rowKey, pageSize = 10 }: ResultsTableProps<T>) {
  const [page, setPage] = useState(0);

  useEffect(() => {
    setPage(0);
  }, [data]);

  const totalPages = Math.max(1, Math.ceil(data.length / pageSize));
  const visibleRows = data.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <div className="results-table-wrap">
      <table className="results-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.label}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row) => (
            <tr key={rowKey(row)}>
              {columns.map((column) => (
                <td key={column.label}>{column.render(row)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="pagination">
        <button onClick={() => setPage(page - 1)} disabled={page === 0}>
          Previous
        </button>
        <span>
          Page {page + 1} of {totalPages}
        </span>
        <button onClick={() => setPage(page + 1)} disabled={page + 1 >= totalPages}>
          Next
        </button>
      </div>
    </div>
  );
}
