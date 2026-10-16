"""
Core domain types shared by every part of the system.

This module is deliberately "pure": it has no heavy dependencies (no Docling,
no Ollama, no Qdrant). Every other module depends on it; it depends on nothing.

It contains three kinds of things:

* Data types - `Chunk` and `SearchHit`, the values that flow through the system.
* Interfaces - `Chunker` and `Embedder` describe WHAT a component does without
  saying HOW. The pipeline is written against these interfaces, so we can swap
  an implementation (e.g. Ollama embeddings -> another provider's embeddings) or use tiny
  fakes in tests without changing a single line of pipeline code.
* Errors     - `DocsGroundTruthError`, the base class for problems we expect and
  can explain to the user (Ollama not running, nothing indexed yet, ...).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Fixed namespace used to generate chunk IDs (see `Chunk.id`).
# Never change it: doing so would give every stored chunk a new ID.
_CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "docsgroundtruth/chunk")


class DocsGroundTruthError(Exception):
    """An expected, user-fixable problem. The CLI shows its message without a traceback."""


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

    def to_payload(self) -> dict[str, Any]:
        """Convert to a plain JSON-friendly dict, stored next to the vector in the database."""
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
class SearchHit:
    """A chunk returned by a search, with its similarity score (higher = more similar)."""

    chunk: Chunk
    score: float


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
    """Turns text into vectors.

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
