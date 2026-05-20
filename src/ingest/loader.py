#!/usr/bin/env python3
"""
loader.py --- corpus loader and batch upsert for the ingestion pipeline

Contains:
    Document: one raw document loaded from a corpus directory
    IngestStats: counts and timing for one ingestion run (post-dedupe)
    CorpusIngestor: chunks, embeds, and indexes documents in batches
    load_corpus(): loads all supported documents from a directory
    load_benchmark_corpus(): loads the 500-doc legal/tech benchmark set
    main(): CLI entrypoint for one-off ingestion runs
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.ingest.bm25_index import BM25Index
from src.ingest.chunking import Chunker
from src.ingest.dense_index import DenseIndex, retry_with_backoff
from src.retrieval.embeddings import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md"}  # everything else is skipped with a warning
JSONL_SUFFIX = ".jsonl"
BENCHMARK_CORPUS_DIR = Path("data/benchmark")  # mounted, not committed


@dataclass(frozen=True)
class Document:
    """Carries one raw document loaded from a corpus directory (text + metadata).

    Attributes:
        doc_id: Stable identifier derived from the file name.
        title: Human-readable document title.
        text: Full document text.
        metadata: Extra attributes carried through to chunks.
    """

    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IngestStats:
    """Carries counts and timing for one ingestion run (deduplicated).

    Attributes:
        documents: Documents ingested after deduplication.
        chunks: Chunks written to the indexes.
        seconds: Wall-clock time the run took.
    """

    documents: int
    chunks: int
    seconds: float


class CorpusIngestor:
    """Chunks, embeds, and indexes documents into the retrieval stores (dense + sparse).

    Attributes:
        chunker: Splitter that turns documents into chunks.
        embedder: Dense embedder for chunk vectors.
        dense_index: Vector store receiving embedded chunks.
        bm25_index: Sparse index receiving chunk text.
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: SentenceTransformerEmbedder,
        dense_index: DenseIndex,
        bm25_index: BM25Index,
        batch_size: int = 100,  # chunks per embed+upsert batch
    ) -> None:
        """Stores the collaborators used during ingestion.

        Args:
            chunker: Splitter that turns documents into chunks.
            embedder: Dense embedder for chunk vectors.
            dense_index: Vector store receiving embedded chunks.
            bm25_index: Sparse index receiving chunk text.
            batch_size: Chunks embedded and upserted per batch.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.batch_size = batch_size

    def ingest(self, documents: list[Document]) -> IngestStats:
        """Ingests documents into the dense and sparse indexes.

        Args:
            documents: Raw documents to chunk, embed, and index.

        Returns:
            stats: Document/chunk counts and timing for the run.
        """
        started = time.perf_counter()
        documents = list({document.doc_id: document for document in documents}.values())
        chunks = [
            chunk
            for document in documents
            for chunk in self.chunker.split(document.text, document.doc_id)
        ]
        logger.info("ingesting %d docs as %d chunks (batch=%d)", len(documents), len(chunks), self.batch_size)
        self.dense_index.ensure_collection()
        total = len(chunks)
        for start in range(0, total, self.batch_size):  # sequential batches keep memory flat
            end = min(start + self.batch_size, total)
            batch = chunks[start:end]
            batch_texts = [chunk.text for chunk in batch]  # embed once, reuse for payload
            vectors = self.embedder.encode(batch_texts)
            retry_with_backoff(
                lambda: self.dense_index.upsert(
                    [chunk.chunk_id for chunk in batch],
                    vectors,
                    [
                        {"doc_id": chunk.doc_id, "text": text}
                        for chunk, text in zip(batch, batch_texts, strict=True)
                    ],
                )
            )
            logger.info("ingested chunks %d-%d of %d", start, end, total)
        self.bm25_index.build(chunks)
        elapsed = time.perf_counter() - started
        logger.info("ingest done: %d chunks in %.1fs", len(chunks), elapsed)
        return IngestStats(documents=len(documents), chunks=len(chunks), seconds=elapsed)


def load_corpus(path: Path) -> list[Document]:
    """Loads all supported documents from a directory (txt/md/jsonl).

    Args:
        path: Directory holding .txt/.md corpus files.

    Returns:
        documents: Loaded documents sorted by file name.
    """
    documents = []
    for file_path in sorted(path.iterdir()):
        if file_path.suffix == JSONL_SUFFIX:
            documents.extend(_load_jsonl(file_path))
            continue
        if file_path.suffix not in TEXT_SUFFIXES:
            continue
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("skipping empty file %s", file_path.name)
            continue
        documents.append(Document(doc_id=file_path.stem, title=file_path.stem, text=text))
    logger.info("loaded %d documents from %s (deduped)", len(documents), path)
    return documents


def _load_jsonl(file_path: Path) -> list[Document]:  # one document per line, blanks skipped
    """Loads documents from one JSONL corpus file.

    Args:
        file_path: JSONL file with doc_id/title/text per line.

    Returns:
        documents: One Document per non-empty line.
    """
    documents = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        documents.append(
            Document(
                doc_id=record["doc_id"],
                title=record.get("title", record["doc_id"]),
                text=record["text"],
                metadata=record.get("metadata", {}),
            )
        )
    return documents


def load_benchmark_corpus(path: Path | None = None) -> list[Document]:
    """Loads the 500-doc legal/tech benchmark corpus used by all eval runs.

    Args:
        path: Corpus directory; defaults to BENCHMARK_CORPUS_DIR.

    Returns:
        documents: Benchmark documents sorted by file name.
    """
    corpus_dir = path or BENCHMARK_CORPUS_DIR
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"benchmark corpus not found at {corpus_dir}; mount it or pass --corpus"
        )
    return load_corpus(corpus_dir)


def main() -> None:
    """Runs a one-off ingestion run from the command line."""
    parser = argparse.ArgumentParser(description="Ingest a corpus into Retrieve-Me")
    parser.add_argument("--corpus", required=True, type=Path, help="corpus directory")
    parser.add_argument("--benchmark", action="store_true", help="ingest the benchmark set")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    documents = (
        load_benchmark_corpus() if args.benchmark else load_corpus(args.corpus)
    )
    stats = _build_ingestor().ingest(documents)
    logger.info(
        "done: %d documents, %d chunks in %.1fs",
        stats.documents,
        stats.chunks,
        stats.seconds,
    )


def _build_ingestor() -> CorpusIngestor:
    """Builds a CorpusIngestor wired to the configured live stores.

    Returns:
        ingestor: Ingestor backed by Qdrant, Redis-free BM25, and the embedder.
    """
    from src.api.dependencies import get_settings, get_qdrant_pool
    from src.ingest.chunking import SemanticClauseChunker
    from src.ingest.dense_index import DenseIndex, QdrantConfig
    from src.retrieval.embeddings import EmbeddingConfig

    settings = get_settings()
    config = QdrantConfig(url=settings.qdrant_url, collection=settings.qdrant_collection)
    dense_index = DenseIndex(config, get_qdrant_pool(settings))
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name=settings.embedding_model))
    return CorpusIngestor(SemanticClauseChunker(), embedder, dense_index, BM25Index())


if __name__ == "__main__":
    main()
