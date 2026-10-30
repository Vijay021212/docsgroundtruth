"""
Shared test fixtures.

Tests must be fast and must run anywhere, including GitHub Actions where there
are no models and no Ollama. So the heavy components are replaced by fakes that
follow the same interfaces (see domain.py):

* Ollama embeddings  -> `FakeEmbedder`: a few lines of hashing
* fastembed BM25     -> `FakeKeywordEncoder`: one slot per distinct word
* fastembed reranker -> `FakeReranker`: the share of question words found in a text
* Qdrant on disk     -> Qdrant in memory: the real database code, just never saved

pytest discovers this file automatically, and any test can request a fixture
just by naming it as a parameter, e.g. `def test_x(store, embedder): ...`.
"""

import hashlib
import math
import re
import zlib
from collections import Counter
from collections.abc import Callable, Iterator

import pytest
from qdrant_client import QdrantClient

from docsgroundtruth.domain import Chunk, SparseVector
from docsgroundtruth.vector_store import QdrantStore


def _words(text: str) -> list[str]:
    """Lower-case words of a text: "What is a HIN?" -> ["what", "is", "a", "hin"]."""
    return re.findall(r"[a-z0-9]+", text.lower())


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


class FakeKeywordEncoder:
    """Stand-in for BM25: one slot per distinct word, weighted by how often it appears."""

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._encode(text) for text in texts]

    def encode_query(self, text: str) -> SparseVector:
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> SparseVector:
        counts = Counter(_words(text))
        return SparseVector(
            # crc32 gives the same number for the same word on every run (unlike hash()).
            indices=tuple(zlib.crc32(word.encode()) for word in counts),
            values=tuple(float(count) for count in counts.values()),
        )


class FakeReranker:
    """Stand-in for the reranker: relevance = share of the question's words found in a text."""

    def score(self, question: str, texts: list[str]) -> list[float]:
        question_words = set(_words(question))
        if not question_words:
            return [0.0] * len(texts)
        return [len(question_words & set(_words(text))) / len(question_words) for text in texts]


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
def keyword_encoder() -> FakeKeywordEncoder:
    return FakeKeywordEncoder()


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    """A fresh, empty in-memory store for each test, closed automatically afterwards."""
    with QdrantStore(QdrantClient(location=":memory:"), collection="test") as test_store:
        yield test_store


@pytest.fixture
def index_chunks(
    store: QdrantStore, embedder: FakeEmbedder, keyword_encoder: FakeKeywordEncoder
) -> Callable[[list[Chunk]], None]:
    """Factory fixture: `index_chunks(chunks)` embeds and stores chunks like the pipeline."""

    def _index(chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        store.ensure_collection(embedder.dimension)
        store.replace_source(
            chunks[0].source,
            chunks,
            embedder.embed_documents(texts),
            keyword_encoder.encode_documents(texts),
        )

    return _index
