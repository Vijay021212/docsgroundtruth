"""
Vector storage and similarity search, backed by Qdrant.

Qdrant stores each chunk as a "point":

    id       -> the chunk's deterministic UUID (see Chunk.id)
    vector   -> the chunk's embedding (768 numbers)
    payload  -> the chunk's text and metadata (source file, pages, headings)

A search means "return the points whose vectors are closest to this query vector".

We run Qdrant in *local mode*: it runs inside our Python process and saves to
a folder (data/index/), so no server or Docker is needed. Local mode compares
the query against every stored vector, which is fine for thousands of chunks.
For millions, the same code can point at a Qdrant server instead; only
`open_local` would change.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from docsgroundtruth.domain import Chunk, DocsGroundTruthError, SearchHit


class QdrantStore:
    """All database access goes through this class: no other module imports qdrant_client."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        # The client is passed in rather than created here ("dependency injection"):
        # the app passes a disk-backed client, tests pass a fast in-memory one.
        self._client = client
        self._collection = collection

    @classmethod
    def open_local(cls, path: Path, collection: str) -> QdrantStore:
        """Open (or create) a store saved in the folder at `path`."""
        path.mkdir(parents=True, exist_ok=True)
        return cls(QdrantClient(path=str(path)), collection)

    # --- Writing -------------------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it doesn't exist yet; otherwise check its vector size.

        A collection is like a table: every vector in it must have the same length.
        If you switch to an embedding model with a different dimension, the old
        index can't be reused, and we say so clearly instead of failing later
        with a cryptic error.
        """
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                # COSINE = the cosine similarity you calculated by hand in Phase 0.
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            return

        existing = self._client.get_collection(self._collection).config.params.vectors.size
        if existing != dimension:
            raise DocsGroundTruthError(
                f"Collection '{self._collection}' stores {existing}-dimensional vectors, "
                f"but the embedding model produces {dimension}. Delete the index folder "
                "and ingest again, or choose another DGT_COLLECTION_NAME."
            )

    def replace_source(self, source: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store a document's chunks, replacing anything previously stored for that document.

        Why delete first instead of just overwriting? If a new version of a file
        produces fewer chunks than before (say 10 instead of 12), overwriting
        would leave the old chunks 11 and 12 behind as stale, wrong answers.
        """
        # Build the points first: zip(strict=True) raises if the lists differ in
        # length, so a bug can never delete old data and then fail to write new data.
        points = [
            PointStruct(id=chunk.id, vector=vector, payload=chunk.to_payload())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=self._source_filter(source)),
        )
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    # --- Reading -------------------------------------------------------------

    def search(self, vector: list[float], top_k: int = 5) -> list[SearchHit]:
        """Return the `top_k` chunks most similar to `vector`, best match first."""
        self._require_collection()
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            with_payload=True,  # we need the text and metadata back, not just IDs
        )
        return [
            SearchHit(chunk=Chunk.from_payload(point.payload or {}), score=point.score)
            for point in response.points
        ]

    def iter_chunks(self, page_size: int = 256) -> Iterator[Chunk]:
        """Yield every stored chunk, in no particular order.

        Qdrant's `scroll` reads the collection one page at a time. Reading in
        pages instead of all at once keeps memory use flat however large the
        index grows. Vectors aren't fetched, since only the payloads are needed.
        """
        self._require_collection()
        offset = None  # None = start from the beginning
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                yield Chunk.from_payload(record.payload or {})
            if offset is None:  # Qdrant returns no next offset after the last page
                return

    def count(self) -> int:
        """Total number of chunks stored (0 if nothing has been ingested yet)."""
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(collection_name=self._collection, exact=True).count

    # --- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Release the index folder. Local mode locks it while it's open."""
        self._client.close()

    # __enter__/__exit__ let callers write `with QdrantStore.open_local(...) as store:`,
    # which guarantees close() runs even if an error happens inside the block.
    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- Helpers -------------------------------------------------------------

    def _require_collection(self) -> None:
        """Raise a friendly error if nothing has been ingested yet (no collection exists)."""
        if not self._client.collection_exists(self._collection):
            raise DocsGroundTruthError(
                "Nothing is indexed yet. Run `docsgroundtruth ingest` first."
            )

    @staticmethod
    def _source_filter(source: str) -> Filter:
        """Database filter matching every chunk whose payload `source` equals `source`."""
        return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
