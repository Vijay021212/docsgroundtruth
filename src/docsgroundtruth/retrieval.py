"""
Retrieval: turn a question into the most relevant chunks.

Four search modes (see SearchMode in domain.py), each building on the one before:

    VECTOR   embed the question, find the chunks closest in MEANING
    KEYWORD  find the chunks with the best BM25 match for the question's WORDS
    HYBRID   run both, then merge the two ranked lists with Reciprocal Rank Fusion
    RERANK   HYBRID, then a reranker reads the question with each candidate and
             re-orders them by how well they actually answer it (the default)

Every interface (CLI, playground, later the API) calls this one module, so they
all behave the same way and all get any improvement made here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from docsgroundtruth.domain import (
    Chunk,
    DocsGroundTruthError,
    Embedder,
    Reranker,
    SearchHit,
    SearchMode,
    SparseEncoder,
)
from docsgroundtruth.timing import StepTimer
from docsgroundtruth.vector_store import QdrantStore

# The constant in Reciprocal Rank Fusion: each list gives a chunk 1 / (RRF_K + position)
# points. 60 is the standard value from the original RRF paper (Cormack et al., 2009).
# It keeps the gap between neighbouring positions small, so no single list dominates.
RRF_K = 60


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The chunks found for one question, and how long each step took."""

    question: str
    mode: SearchMode
    hits: tuple[SearchHit, ...]  # best match first
    # Seconds per step, in the order the steps ran, e.g. {"question embedding": 0.31, ...}.
    # A dict rather than fixed fields, because each search mode runs different steps.
    timings: dict[str, float]

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def summary(self) -> str:
        """One line for any interface, e.g. "5 results in 0.35s (question embedding 0.30s, ...)"."""
        steps = ", ".join(f"{name} {seconds:.2f}s" for name, seconds in self.timings.items())
        return f"{len(self.hits)} results in {self.total_seconds:.2f}s ({steps})"


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[SearchHit]], limit: int
) -> list[SearchHit]:
    """Merge several ranked lists into one, using positions only, never raw scores.

    Each list gives a chunk 1 / (RRF_K + position) points, and the points are
    added up, so a chunk found high in several lists collects the most. Raw
    scores can't be combined directly: they're on different scales (a BM25 score
    of 12.7 vs a cosine similarity of 0.78).

    Each returned hit's `score` is its total points, and its `ranks` record where
    each list placed it, e.g. {"vector": 3, "keyword": 1}.
    """
    points: dict[str, float] = defaultdict(float)  # chunk id -> total points
    ranks: dict[str, dict[str, int]] = defaultdict(dict)  # chunk id -> {list name: position}
    chunks: dict[str, Chunk] = {}  # chunk id -> the chunk itself

    for list_name, hits in ranked_lists.items():
        for position, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.id
            points[chunk_id] += 1 / (RRF_K + position)
            ranks[chunk_id][list_name] = position
            chunks[chunk_id] = hit.chunk

    best_first = sorted(points, key=lambda chunk_id: points[chunk_id], reverse=True)
    return [
        SearchHit(chunk=chunks[chunk_id], score=points[chunk_id], ranks=ranks[chunk_id])
        for chunk_id in best_first[:limit]
    ]


class Retriever:
    """Finds the chunks most relevant to a question."""

    def __init__(
        self,
        embedder: Embedder,
        keyword_encoder: SparseEncoder,
        reranker: Reranker,
        store: QdrantStore,
        candidates: int = 30,
    ) -> None:
        """
        Args:
            candidates: How many results each search hands on to fusion and the
                reranker in the HYBRID and RERANK modes.
        """
        self._embedder = embedder
        self._keyword_encoder = keyword_encoder
        self._reranker = reranker
        self._store = store
        self._candidates = candidates

    def retrieve(
        self, question: str, top_k: int = 5, mode: SearchMode = SearchMode.RERANK
    ) -> RetrievalResult:
        """Return the `top_k` chunks most relevant to `question`, best match first."""
        question = question.strip()
        if not question:
            raise DocsGroundTruthError("Please enter a question.")
        if top_k < 1:
            raise DocsGroundTruthError("The number of results must be at least 1.")

        timer = StepTimer()
        if mode is SearchMode.VECTOR:
            hits = self._vector_search(question, top_k, timer)
        elif mode is SearchMode.KEYWORD:
            hits = self._keyword_search(question, top_k, timer)
        else:
            hits = self._hybrid_search(question, top_k, timer, rerank=mode is SearchMode.RERANK)
        return RetrievalResult(
            question=question, mode=mode, hits=tuple(hits), timings=timer.timings
        )

    def _vector_search(self, question: str, limit: int, timer: StepTimer) -> list[SearchHit]:
        """Meaning-based search: embed the question, then find the nearest chunk vectors."""
        with timer.step("question embedding"):
            # The same model as the chunks, but with the query prefix (see OllamaEmbedder).
            vector = self._embedder.embed_query(question)
        with timer.step("vector search"):
            return self._store.vector_search(vector, top_k=limit)

    def _keyword_search(self, question: str, limit: int, timer: StepTimer) -> list[SearchHit]:
        """Word-based search: BM25 over the question's words."""
        with timer.step("keyword search"):
            # Encoding the question is only word clean-up and hashing: microseconds.
            vector = self._keyword_encoder.encode_query(question)
            return self._store.keyword_search(vector, top_k=limit)

    def _hybrid_search(
        self, question: str, top_k: int, timer: StepTimer, rerank: bool
    ) -> list[SearchHit]:
        """Both searches, merged with RRF, then optionally re-checked by the reranker."""
        # Each search returns more than the final top_k, so fusion and the
        # reranker have a wide pool of candidates to choose from.
        pool = max(top_k, self._candidates)
        ranked_lists = {
            "vector": self._vector_search(question, pool, timer),
            "keyword": self._keyword_search(question, pool, timer),
        }
        with timer.step("fusion"):
            hits = reciprocal_rank_fusion(ranked_lists, limit=pool)
        if rerank:
            with timer.step("reranking"):
                hits = self._rerank(question, hits)
        return hits[:top_k]

    def _rerank(self, question: str, candidates: list[SearchHit]) -> list[SearchHit]:
        """Re-score the candidates with the reranker, then sort them by its relevance score."""
        if not candidates:
            return []
        scores = self._reranker.score(question, [hit.chunk.text for hit in candidates])
        # replace() copies a frozen hit with a new score. Its `ranks` are kept, so
        # interfaces can still show where each search had placed the chunk.
        rescored = [
            replace(hit, score=score) for hit, score in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)
