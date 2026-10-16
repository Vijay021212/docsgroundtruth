"""
The ingestion pipeline:   file  ->  chunks  ->  vectors  ->  vector store

This module only ORCHESTRATES. It doesn't know about Docling or Ollama: it
calls them through the `Chunker` and `Embedder` interfaces (see domain.py).
That's what keeps each component replaceable and testable on its own.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from docsgroundtruth.domain import Chunk, Chunker, DocsGroundTruthError, Embedder
from docsgroundtruth.timing import Stopwatch
from docsgroundtruth.vector_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What happened to one file during ingestion."""

    source: str
    chunk_count: int
    seconds: float


class IngestionPipeline:
    """Turns documents on disk into searchable chunks in the vector store."""

    def __init__(self, chunker: Chunker, embedder: Embedder, store: QdrantStore) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def ingest_path(self, path: Path) -> list[IngestReport]:
        """Ingest one file, or every supported file directly inside a folder."""
        files = self._find_documents(path)

        # Prepare the collection BEFORE the slow parsing step, so a problem
        # (Ollama not running, wrong vector size) shows up in seconds rather
        # than after minutes of PDF processing.
        self._store.ensure_collection(self._embedder.dimension)

        return [self._ingest_file(file) for file in files]

    def _ingest_file(self, path: Path) -> IngestReport:
        """Run one file through every stage, logging how long each stage took."""
        with Stopwatch() as total:
            # Stages 1-2 (parse, chunk): the chunker times and logs these itself.
            chunks = self._chunker.chunk_file(path)

            # Stage 3: embed.
            vectors = self._embed(chunks)

            # Stage 4: store. Runs even when there are no chunks, so stale chunks
            # from an older version of the file are still removed.
            with Stopwatch() as watch:
                self._store.replace_source(path.name, chunks, vectors)
            logger.info("Stored %d chunks in %.1fs", len(chunks), watch.seconds)

        report = IngestReport(path.name, len(chunks), total.seconds)
        if chunks:
            logger.info(
                "Ingested %s: %d chunks in %.1fs", report.source, report.chunk_count, report.seconds
            )
        else:
            logger.warning(
                "No text found in %s. If it is a scanned document, make sure OCR is on "
                "(DGT_ENABLE_OCR=true).",
                path.name,
            )
        return report

    def _embed(self, chunks: list[Chunk]) -> list[list[float]]:
        """Turn the chunks' text into vectors (stage 3), logging how long it took."""
        if not chunks:
            return []
        with Stopwatch() as watch:
            vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        logger.info("Embedded %d chunks in %.1fs", len(chunks), watch.seconds)
        return vectors

    def _find_documents(self, path: Path) -> list[Path]:
        """Resolve `path` to the list of files to ingest, keeping only supported formats."""
        if not path.exists():
            raise DocsGroundTruthError(f"Path not found: {path}")

        candidates = [path] if path.is_file() else sorted(path.iterdir())
        documents = [
            file
            for file in candidates
            if file.is_file() and file.suffix.lower() in self._chunker.supported_suffixes
        ]
        if not documents:
            supported = ", ".join(sorted(self._chunker.supported_suffixes))
            raise DocsGroundTruthError(f"No supported documents ({supported}) found at {path}")
        return documents
