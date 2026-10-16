"""Tests for IngestionPipeline, using fakes instead of Docling and Ollama."""

from pathlib import Path

import pytest

from docsgroundtruth.domain import Chunk, DocsGroundTruthError
from docsgroundtruth.pipeline import IngestionPipeline


class FakeChunker:
    """Stand-in for DoclingChunker: returns pre-defined chunk texts for each file name."""

    supported_suffixes = frozenset({".pdf"})

    def __init__(self, make_chunks, texts_by_file: dict[str, list[str]]) -> None:
        self._make_chunks = make_chunks
        self._texts_by_file = texts_by_file

    def chunk_file(self, path: Path) -> list[Chunk]:
        return self._make_chunks(path.name, self._texts_by_file.get(path.name, []))


@pytest.fixture
def documents_dir(tmp_path: Path) -> Path:
    """A temporary folder with two PDFs and one unsupported file. Contents don't matter."""
    for name in ("a.pdf", "b.pdf", "notes.txt"):
        (tmp_path / name).touch()
    return tmp_path


def test_ingests_every_supported_file_in_a_folder(documents_dir, store, embedder, make_chunks):
    chunker = FakeChunker(make_chunks, {"a.pdf": ["one", "two"], "b.pdf": ["three"]})
    pipeline = IngestionPipeline(chunker, embedder, store)

    reports = pipeline.ingest_path(documents_dir)

    assert [(r.source, r.chunk_count) for r in reports] == [("a.pdf", 2), ("b.pdf", 1)]
    assert store.count() == 3  # notes.txt was skipped


def test_ingests_a_single_file(documents_dir, store, embedder, make_chunks):
    chunker = FakeChunker(make_chunks, {"a.pdf": ["one", "two"]})

    reports = IngestionPipeline(chunker, embedder, store).ingest_path(documents_dir / "a.pdf")

    assert [r.source for r in reports] == ["a.pdf"]


def test_missing_path_raises_a_clear_error(tmp_path, store, embedder, make_chunks):
    pipeline = IngestionPipeline(FakeChunker(make_chunks, {}), embedder, store)

    with pytest.raises(DocsGroundTruthError, match="not found"):
        pipeline.ingest_path(tmp_path / "does-not-exist")


def test_folder_without_supported_files_raises(tmp_path, store, embedder, make_chunks):
    (tmp_path / "notes.txt").touch()
    pipeline = IngestionPipeline(FakeChunker(make_chunks, {}), embedder, store)

    with pytest.raises(DocsGroundTruthError, match="No supported documents"):
        pipeline.ingest_path(tmp_path)
