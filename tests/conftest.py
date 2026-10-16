"""
Shared test fixtures.

Tests must be fast and must run anywhere, including GitHub Actions where there
is no Ollama and no downloaded model. So the heavy components are replaced:

* Ollama -> `FakeEmbedder`, a few lines of hashing that follow the same `Embedder` interface.
* Qdrant on disk -> Qdrant in memory: the real database code, just never written to disk.

pytest discovers this file automatically, and any test can request a fixture
just by naming it as a parameter, e.g. `def test_x(store, embedder): ...`.
"""

import hashlib
import math
from collections.abc import Callable, Iterator

import pytest
from qdrant_client import QdrantClient

from docsgroundtruth.domain import Chunk
from docsgroundtruth.vector_store import QdrantStore


class FakeEmbedder:
    """Deterministic stand-in for Ollama: the same text always gives the same unit vector.

    The vectors carry no meaning; they only need to be stable and distinct,
    which is enough to test storage, replacement and search ordering.
    """

    dimension = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        # sha256 always gives the same bytes for the same text, so tests repeat
        # exactly. (Python's built-in hash() is randomised per run, so we avoid it.)
        digest = hashlib.sha256(text.encode()).digest()
        raw = [byte - 127.5 for byte in digest[: self.dimension]]
        norm = math.sqrt(sum(value * value for value in raw))
        return [value / norm for value in raw]


def _make_chunks(source: str, texts: list[str]) -> list[Chunk]:
    """Build one chunk per text, numbered in order, all on page 1."""
    return [Chunk(source=source, index=i, text=text, pages=(1,)) for i, text in enumerate(texts)]


@pytest.fixture
def make_chunks() -> Callable[[str, list[str]], list[Chunk]]:
    """Factory fixture: call `make_chunks("file.pdf", ["text a", "text b"])` inside a test."""
    return _make_chunks


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    """A fresh, empty in-memory store for each test, closed automatically afterwards."""
    with QdrantStore(QdrantClient(location=":memory:"), collection="test") as test_store:
        yield test_store


@pytest.fixture
def index_chunks(store: QdrantStore, embedder: FakeEmbedder) -> Callable[[list[Chunk]], None]:
    """Factory fixture: `index_chunks(chunks)` embeds and stores chunks like the pipeline."""

    def _index(chunks: list[Chunk]) -> None:
        store.ensure_collection(embedder.dimension)
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        store.replace_source(chunks[0].source, chunks, vectors)

    return _index
