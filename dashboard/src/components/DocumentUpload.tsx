/**
 * DocumentUpload.tsx --- drop zone for adding documents to the index
 *
 * Contains:
 *   DocumentUploadProps: upload handler and the state it reports back
 *   DocumentUpload: drop zone, file picker, and the result of the last upload
 */

import { useRef, useState } from "react";

import type { UploadResult } from "../types";

export interface DocumentUploadProps {
  onUpload: (files: File[]) => void;
  result: UploadResult | null;
  isUploading: boolean;
  error: string | null;
}

/**
 * Renders the area a user drops documents onto to index them.
 *
 * @param props - Upload handler plus the state of the last upload.
 * @returns element - Upload panel element.
 */
export function DocumentUpload({ onUpload, result, isUploading, error }: DocumentUploadProps) {
  const [isOver, setIsOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = (list: FileList | null) => {
    const files = Array.from(list ?? []);
    if (files.length > 0) {
      onUpload(files);
    }
  };

  return (
    <section className="panel" aria-label="document upload">
      <h2>Add documents</h2>
      <p className="panel-subtitle">
        Drop .txt or .md files here to chunk, embed and index them. They are searchable in the query
        inspector as soon as the upload finishes.
      </p>

      <div
        className={`dropzone${isOver ? " is-over" : ""}${isUploading ? " is-busy" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setIsOver(true);
        }}
        onDragLeave={() => setIsOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsOver(false);
          submit(event.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            inputRef.current?.click();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".txt,.md"
          aria-label="documents to index"
          onChange={(event) => submit(event.target.files)}
          hidden
        />
        <span className="dropzone-title">{isUploading ? "Indexing…" : "Drop documents here"}</span>
        <span className="dropzone-hint">or click to choose files</span>
      </div>

      {error !== null && <p className="error-banner">Upload failed: {error}</p>}

      {result !== null && (
        <div className="upload-result">
          <p className="status-line">
            Indexed <strong>{result.documents.length}</strong>{" "}
            {result.documents.length === 1 ? "document" : "documents"} into{" "}
            <strong>{result.chunks}</strong> chunks in {result.tookMs} ms.
          </p>
          <ul className="upload-list">
            {result.documents.map((doc) => (
              <li key={doc.docId}>
                <span className="upload-name">{doc.title}</span>
                <span className="upload-size">{Math.max(1, Math.round(doc.bytes / 1024))} KB</span>
              </li>
            ))}
          </ul>
          {Object.keys(result.skipped).length > 0 && (
            <ul className="upload-list upload-skipped">
              {Object.entries(result.skipped).map(([name, reason]) => (
                <li key={name}>
                  <span className="upload-name">{name}</span>
                  <span className="upload-size">skipped — {reason}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
