#!/usr/bin/env ts-node
/**
 * ResultsTable.tsx --- generic paginated table for dashboard data
 *
 * Contains:
 *   ResultsTable: renders one page of rows with prev/next pagination
 */

import { useEffect, useMemo, useState } from "react";

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
export function ResultsTable<T>({ data, columns, rowKey, pageSize = 25 }: ResultsTableProps<T>) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(pageSize);

  useEffect(() => {
    setPage(0);
  }, [data, rowsPerPage]);

  const totalPages = Math.max(1, Math.ceil(data.length / rowsPerPage));

  useEffect(() => {
    if (page >= totalPages) {
      setPage(Math.max(0, totalPages - 1));
    }
  }, [page, totalPages]);

  const visibleRows = useMemo(
    () => data.slice(page * rowsPerPage, (page + 1) * rowsPerPage),
    [data, page, rowsPerPage],
  );

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowLeft" || event.key === "PageUp" && page > 0) {
      setPage(page - 1);
    }
    if (event.key === "ArrowRight" || event.key === "PageDown" && page + 1 < totalPages) {
      setPage(page + 1);
    }
  };

  return (
    <div className="results-table-wrap" onKeyDown={handleKeyDown} tabIndex={0}
    role="group"
    aria-label="results pagination">
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
        <select
          aria-label="results per page"
          value={rowsPerPage}
          onChange={(event) => setRowsPerPage(Number(event.target.value))}
        >
          {[5, 10, 25, 50].map((size) => (
            <option key={size} value={size}>
              {size} / page
            </option>
          ))}
        </select>
        <button aria-label="previous page" onClick={() => setPage(page - 1)} disabled={page === 0}>
          Previous
        </button>
        <span aria-current="page">
          Page {page + 1} of {totalPages}
        </span>
        <button aria-label="next page" onClick={() => setPage(page + 1)} disabled={page + 1 >= totalPages}>
          Next
        </button>
      </div>
    </div>
  );
}
