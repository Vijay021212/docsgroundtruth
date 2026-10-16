"""
Retrieval: turn a question into the most relevant chunks.

Today this is pure vector search: embed the question, then find the chunks
whose vectors are nearest. Phase 3 adds keyword search (BM25), result fusion
and a reranker HERE, so every interface (CLI, playground, later the API) gets
the improvements without any change on its side.
"""

from __future__ import annotations

from dataclasses import dataclass

from docsgroundtruth.domain import DocsGroundTruthError, Embedder, SearchHit
from docsgroundtruth.timing import Stopwatch
from docsgroundtruth.vector_store import QdrantStore


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The chunks found for one question, and how long each step took."""

    question: str
    hits: tuple[SearchHit, ...]  # best match first
    # Seconds per step, in the order the steps ran, e.g. {"question embedding": 0.31, ...}.
    # A dict rather than fixed fields, so Phase 3 can add steps without changing this class.
    timings: dict[str, float]

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def summary(self) -> str:
        """One line for any interface, e.g. "5 results in 0.35s (question embedding 0.30s, ...)"."""
        steps = ", ".join(f"{name} {seconds:.2f}s" for name, seconds in self.timings.items())
        return f"{len(self.hits)} results in {self.total_seconds:.2f}s ({steps})"


class Retriever:
    """Finds the chunks most relevant to a question."""

    def __init__(self, embedder: Embedder, store: QdrantStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, top_k: int = 5) -> RetrievalResult:
        """Return the `top_k` chunks most relevant to `question`, best match first."""
        question = question.strip()
        if not question:
            raise DocsGroundTruthError("Please enter a question.")
        if top_k < 1:
            raise DocsGroundTruthError("The number of results must be at least 1.")

        timings: dict[str, float] = {}

        # Step 1: turn the question into a vector, the same way chunks were embedded
        # (but with the query prefix, see OllamaEmbedder.embed_query).
        with Stopwatch() as watch:
            vector = self._embedder.embed_query(question)
        timings["question embedding"] = watch.seconds

        # Step 2: find the stored chunks whose vectors are closest to it.
        with Stopwatch() as watch:
            hits = self._store.search(vector, top_k=top_k)
        timings["vector search"] = watch.seconds

        return RetrievalResult(question=question, hits=tuple(hits), timings=timings)
