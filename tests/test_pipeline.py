"""Tests for IngestionPipeline, using fakes instead of Docling, Ollama and fastembed."""

from collections.abc import Callable
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
def make_pipeline(
    store, embedder, keyword_encoder, make_chunks
) -> Callable[[dict[str, list[str]]], IngestionPipeline]:
    """Factory fixture: `make_pipeline({"a.pdf": ["text", ...]})` builds a pipeline of fakes."""

    def _make(texts_by_file: dict[str, list[str]]) -> IngestionPipeline:
        chunker = FakeChunker(make_chunks, texts_by_file)
        return IngestionPipeline(chunker, embedder, keyword_encoder, store)

    return _make


@pytest.fixture
def documents_dir(tmp_path: Path) -> Path:
    """A temporary folder with two PDFs and one unsupported file. Contents don't matter."""
    for name in ("a.pdf", "b.pdf", "notes.txt"):
        (tmp_path / name).touch()
    return tmp_path


def test_ingests_every_supported_file_in_a_folder(documents_dir, store, make_pipeline):
    pipeline = make_pipeline({"a.pdf": ["one", "two"], "b.pdf": ["three"]})

    reports = pipeline.ingest_path(documents_dir)

    assert [(r.source, r.chunk_count) for r in reports] == [("a.pdf", 2), ("b.pdf", 1)]
    assert store.count() == 3  # notes.txt was skipped


def test_ingests_a_single_file(documents_dir, make_pipeline):
    reports = make_pipeline({"a.pdf": ["one", "two"]}).ingest_path(documents_dir / "a.pdf")

    assert [r.source for r in reports] == ["a.pdf"]


def test_ingested_chunks_are_searchable_by_keyword(
    documents_dir, store, keyword_encoder, make_pipeline
):
    texts = ["the HIN links providers", "claims are paid monthly"]
    make_pipeline({"a.pdf": texts}).ingest_path(documents_dir / "a.pdf")

    hits = store.keyword_search(keyword_encoder.encode_query("HIN"), top_k=1)

    assert hits[0].chunk.text == "the HIN links providers"


def test_missing_path_raises_a_clear_error(tmp_path, make_pipeline):
    with pytest.raises(DocsGroundTruthError, match="not found"):
        make_pipeline({}).ingest_path(tmp_path / "does-not-exist")


def test_folder_without_supported_files_raises(tmp_path, make_pipeline):
    (tmp_path / "notes.txt").touch()

    with pytest.raises(DocsGroundTruthError, match="No supported documents"):
        make_pipeline({}).ingest_path(tmp_path)
