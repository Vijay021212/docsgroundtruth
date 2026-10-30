"""
Vector storage and search, backed by Qdrant.

Qdrant stores each chunk as a "point":

    id       -> the chunk's deterministic UUID (see Chunk.id)
    vectors  -> two per chunk:
                  "dense" = 768 numbers capturing its meaning (embedding model)
                  "bm25"  = its keyword weights, a sparse vector (see bm25.py)
    payload  -> the chunk's text and metadata (source file, pages, headings)

Two searches use them:

    vector_search   "which chunks MEAN something similar?"   cosine similarity on "dense"
    keyword_search  "which chunks contain the same WORDS?"   BM25 on "bm25"

We run Qdrant in *local mode*: it runs inside our Python process and saves to
a folder (data/index/), so no server or Docker is needed. Local mode compares
the query against every stored point, which is fine for thousands of chunks.
For millions, the same code can point at a Qdrant server instead; only
`open_local` would change.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from qdrant_client import QdrantClient, models

from docsgroundtruth.domain import Chunk, DocsGroundTruthError, SearchHit, SparseVector

# Names of the two vectors stored for every chunk.
_DENSE = "dense"
_SPARSE = "bm25"

_OUTDATED_INDEX = (
    "The index was built by an older version of DocsGroundTruth, before keyword search "
    "existed. Delete the index folder (data/index) and run `docsgroundtruth ingest` again."
)


class QdrantStore:
    """All database access goes through this class: no other module imports qdrant_client."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        # The client is passed in rather than created here ("dependency injection"):
        # the app passes a disk-backed client, tests pass a fast in-memory one.
        self._client = client
        self._collection = collection

    @classmethod
    def open_local(cls, path: Path, collection: str) -> QdrantStore:
        """Open (or create) a store saved in the folder at `path`.

        Local mode lets only ONE program open the folder at a time. If another
        program has it open (an ingest run, or the playground mid-search), we
        explain that clearly instead of showing Qdrant's technical error.
        """
        path.mkdir(parents=True, exist_ok=True)
        try:
            client = QdrantClient(path=str(path))
        except RuntimeError as error:
            # Qdrant reports a locked folder as a plain RuntimeError, so its
            # message is the only way to tell this case apart from real failures.
            if "already accessed" not in str(error):
                raise
            raise DocsGroundTruthError(
                f"The index at {path} is in use by another program (an ingest run or the "
                "playground). Wait for it to finish, then try again."
            ) from error
        return cls(client, collection)

    # --- Writing -------------------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it doesn't exist yet; otherwise check its layout and size.

        A collection is like a table, and ours stores two vectors per chunk (see
        the module docstring). The keyword index uses Qdrant's IDF modifier: Qdrant
        counts how many chunks contain each word and adds BM25's "rare words count
        more" part at search time.
        """
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                # COSINE = the cosine similarity you calculated by hand in Phase 0.
                vectors_config={
                    _DENSE: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    _SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            return

        existing = self._check_layout().size
        if existing != dimension:
            raise DocsGroundTruthError(
                f"Collection '{self._collection}' stores {existing}-dimensional vectors, "
                f"but the embedding model produces {dimension}. Delete the index folder "
                "and ingest again, or choose another DGT_COLLECTION_NAME."
            )

    def replace_source(
        self,
        source: str,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVector],
    ) -> None:
        """Store a document's chunks, replacing anything previously stored for that document.

        Why delete first instead of just overwriting? If a new version of a file
        produces fewer chunks than before (say 10 instead of 12), overwriting
        would leave the old chunks 11 and 12 behind as stale, wrong answers.
        """
        # Build the points first: zip(strict=True) raises if the lists differ in
        # length, so a bug can never delete old data and then fail to write new data.
        points = [
            models.PointStruct(
                id=chunk.id,
                vector=self._point_vectors(dense_vector, sparse_vector),
                payload=chunk.to_payload(),
            )
            for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=self._source_filter(source)),
        )
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    # --- Searching -------------------------------------------------------------

    def vector_search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        """The `top_k` chunks whose meaning is closest to `vector` (cosine), best first."""
        return self._query(vector, using=_DENSE, top_k=top_k)

    def keyword_search(self, vector: SparseVector, top_k: int) -> list[SearchHit]:
        """The `top_k` chunks with the best BM25 match for the question's words, best first."""
        if vector.is_empty:
            return []  # the question had only filler words, so there's nothing to match
        return self._query(self._to_qdrant(vector), using=_SPARSE, top_k=top_k)

    # --- Reading -------------------------------------------------------------

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

    def _query(
        self, query: list[float] | models.SparseVector, using: str, top_k: int
    ) -> list[SearchHit]:
        """Search one of the two named vectors (`using`). Shared by both searches."""
        self._require_collection()
        self._check_layout()
        response = self._client.query_points(
            collection_name=self._collection,
            query=query,
            using=using,
            limit=top_k,
            with_payload=True,  # we need the text and metadata back, not just IDs
        )
        return [
            SearchHit(chunk=Chunk.from_payload(point.payload or {}), score=point.score)
            for point in response.points
        ]

    def _require_collection(self) -> None:
        """Raise a friendly error if nothing has been ingested yet (no collection exists)."""
        if not self._client.collection_exists(self._collection):
            raise DocsGroundTruthError(
                "Nothing is indexed yet. Run `docsgroundtruth ingest` first."
            )

    def _check_layout(self) -> models.VectorParams:
        """Check the collection has both named vectors, and return the dense vector's settings.

        Collections created before Phase 3 stored a single unnamed vector. They
        can't do keyword search, so we explain how to rebuild them.
        """
        params = self._client.get_collection(self._collection).config.params
        vectors = params.vectors
        has_dense = isinstance(vectors, dict) and _DENSE in vectors
        has_sparse = _SPARSE in (params.sparse_vectors or {})
        if not (has_dense and has_sparse):
            raise DocsGroundTruthError(_OUTDATED_INDEX)
        return vectors[_DENSE]

    @classmethod
    def _point_vectors(
        cls, dense: list[float], sparse: SparseVector
    ) -> dict[str, list[float] | models.SparseVector]:
        """Both named vectors for one point.

        A chunk with no usable words (e.g. only numbers) gets no keyword vector,
        so keyword search simply never matches it.
        """
        vectors: dict[str, list[float] | models.SparseVector] = {_DENSE: dense}
        if not sparse.is_empty:
            vectors[_SPARSE] = cls._to_qdrant(sparse)
        return vectors

    @staticmethod
    def _to_qdrant(vector: SparseVector) -> models.SparseVector:
        """Convert our SparseVector into Qdrant's own type."""
        return models.SparseVector(indices=list(vector.indices), values=list(vector.values))

    @staticmethod
    def _source_filter(source: str) -> models.Filter:
        """Database filter matching every chunk whose payload `source` equals `source`."""
        return models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
        )
