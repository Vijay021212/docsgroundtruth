"""
Core domain types shared by every part of the system.

This module is deliberately "pure": it has no heavy dependencies (no Docling,
no Ollama, no Qdrant, no fastembed). Every other module depends on it; it
depends on nothing.

It contains four kinds of things:

* Data types   - `Chunk`, `SparseVector` and `SearchHit`, the values that flow
  through the system.
* Search modes - `SearchMode`, the four ways retrieval can search.
* Interfaces   - `Chunker`, `Embedder`, `SparseEncoder`, `Reranker` and `ChatModel`
  describe WHAT a component does without saying HOW. The rest of the code is
  written against these interfaces, so we can swap an implementation (e.g.
  Ollama -> another provider) or use tiny fakes in tests without changing it.
* Errors       - `DocsGroundTruthError`, the base class for problems we expect and
  can explain to the user (Ollama not running, nothing indexed yet, ...).
"""

from __future__ import annotations

import textwrap
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

# Fixed namespace used to generate chunk IDs (see `Chunk.id`).
# Never change it: doing so would give every stored chunk a new ID.
_CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "docsgroundtruth/chunk")


class DocsGroundTruthError(Exception):
    """An expected, user-fixable problem. The CLI shows its message without a traceback."""


# --- Data types --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable piece of a document.

    `frozen=True` makes a chunk immutable: once created, no later step can
    silently change it. `slots=True` makes each instance smaller in memory.
    """

    source: str  # file the chunk came from, e.g. "AHM_250_Transcript.pdf"
    index: int  # position of the chunk inside that file: 0, 1, 2, ...
    text: str  # what gets embedded and shown to the LLM (section headings included)
    pages: tuple[int, ...] = ()  # page numbers the text appears on, used for citations
    headings: tuple[str, ...] = ()  # section path, e.g. ("Chapter 2", "Claims")
    token_count: int = 0  # size of `text` in tokens, handy for checking chunk sizes

    @property
    def id(self) -> str:
        """Deterministic ID: the same source + index always produces the same UUID.

        This is what makes ingestion idempotent (safe to repeat): re-ingesting
        a file writes its chunks under the same IDs instead of adding duplicates.
        """
        return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, f"{self.source}#{self.index}"))

    @property
    def pages_label(self) -> str:
        """Page numbers as display text, e.g. "308, 309" ("?" when the format has no pages)."""
        return ", ".join(map(str, self.pages)) or "?"

    def preview(self, max_chars: int = 300) -> str:
        """The text cut to about `max_chars` characters, on one line, for result lists."""
        # shorten() also collapses newlines and repeated spaces into single spaces.
        return textwrap.shorten(self.text, max_chars, placeholder=" ...")

    def to_payload(self) -> dict[str, Any]:
        """Convert to a plain JSON-friendly dict, stored next to the vectors in the database."""
        return {
            "source": self.source,
            "index": self.index,
            "text": self.text,
            "pages": list(self.pages),
            "headings": list(self.headings),
            "token_count": self.token_count,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Chunk:
        """Rebuild a Chunk from a stored payload (the reverse of `to_payload`)."""
        return cls(
            source=payload["source"],
            index=payload["index"],
            text=payload["text"],
            pages=tuple(payload.get("pages", ())),
            headings=tuple(payload.get("headings", ())),
            token_count=payload.get("token_count", 0),
        )


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A mostly-empty vector, stored as only its non-empty slots.

    Used for BM25 keyword search. Picture one slot for every possible word:
    only the words that appear in a text get a weight, and all the other slots
    are zero. Storing just the few (slot number, weight) pairs keeps it small.
    """

    indices: tuple[int, ...] = ()  # which slots are filled (each number stands for one word)
    values: tuple[float, ...] = ()  # the weight in each of those slots, in the same order

    @property
    def is_empty(self) -> bool:
        """True when the text had no usable words (e.g. only filler words like "what is")."""
        return not self.indices


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A chunk returned by a search, with its score.

    What the score means depends on how the search was done; see
    `SearchMode.score_name`.
    """

    chunk: Chunk
    score: float
    # Position of the chunk in each search that found it, e.g. {"vector": 3, "keyword": 1}.
    # Filled in by hybrid search, so interfaces can explain WHY a chunk ranked where it did.
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def ranks_label(self) -> str:
        """Where each search placed the chunk, e.g. "vector #3 · keyword #1" ("" if unknown)."""
        return " · ".join(f"{name} #{position}" for name, position in self.ranks.items())


# --- Search modes --------------------------------------------------------------------


class SearchMode(StrEnum):
    """The four ways retrieval can search. Each one builds on the one before it.

    A StrEnum member behaves like its string value, so settings and command-line
    options can simply say "hybrid".
    """

    VECTOR = "vector"  # meaning only (the Phase 2 behaviour)
    KEYWORD = "keyword"  # exact words only (BM25)
    HYBRID = "hybrid"  # both searches, merged with Reciprocal Rank Fusion
    RERANK = "rerank"  # hybrid, then a reranker re-checks the best candidates

    @property
    def label(self) -> str:
        """Name shown in interfaces, e.g. "Hybrid + reranker"."""
        return _MODE_DESCRIPTIONS[self][0]

    @property
    def score_name(self) -> str:
        """What a result's score means in this mode, e.g. "relevance"."""
        return _MODE_DESCRIPTIONS[self][1]


_MODE_DESCRIPTIONS: dict[SearchMode, tuple[str, str]] = {
    # mode: (label shown in interfaces, what the score means)
    SearchMode.VECTOR: ("Vector (meaning)", "similarity"),
    SearchMode.KEYWORD: ("Keyword (BM25)", "BM25 score"),
    SearchMode.HYBRID: ("Hybrid (vector + keyword)", "fusion points"),
    SearchMode.RERANK: ("Hybrid + reranker", "relevance"),
}


# --- Interfaces ----------------------------------------------------------------------


class Chunker(Protocol):
    """Turns a document file into chunks.

    A `Protocol` is a structural interface: any class with these members counts
    as a Chunker. It doesn't need to inherit from anything.
    """

    # File extensions this chunker can read, lower-case with the dot, e.g. {".pdf"}.
    supported_suffixes: frozenset[str]

    def chunk_file(self, path: Path) -> list[Chunk]:
        """Parse the file at `path` and split it into chunks."""
        ...


class Embedder(Protocol):
    """Turns text into dense vectors that capture its meaning.

    Documents and queries have separate methods because some models
    (like nomic-embed-text) encode them slightly differently.
    """

    @property
    def dimension(self) -> int:
        """Length of every vector this embedder returns (768 for nomic-embed-text)."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many document chunks; returns one vector per text, in the same order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search question."""
        ...


class SparseEncoder(Protocol):
    """Turns text into sparse keyword vectors (BM25 word weights)."""

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode many chunks; returns one sparse vector per text, in the same order."""
        ...

    def encode_query(self, text: str) -> SparseVector:
        """Encode a single search question."""
        ...


class Reranker(Protocol):
    """Judges how well each text answers a question, by reading the two together."""

    def score(self, question: str, texts: list[str]) -> list[float]:
        """Relevance of each text, from 0 (unrelated) to 1 (answers it), in the same order."""
        ...


class ChatModel(Protocol):
    """Writes text in reply to instructions and a message."""

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the reply piece by piece. `system` holds the rules, `user` the message."""
        ...
