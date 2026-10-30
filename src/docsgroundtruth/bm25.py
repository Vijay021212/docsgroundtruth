"""
BM25 keyword encoding, powered by fastembed.

BM25 scores a chunk by the question words it contains: rare words count more,
repeated mentions count with diminishing returns, and long chunks are adjusted
for their length. The formula is split between two tools:

* fastembed (this module) cleans the text (drops filler words like "what" and
  "is", cuts words to their root: "approvals" -> "approv") and computes each
  chunk's word weights: the mentions-and-length part of the formula.
* Qdrant adds the rarity part (IDF) at search time, from how many chunks contain
  each word. That's why the collection is created with Modifier.IDF
  (see vector_store.py).

The output is a SparseVector: one slot per possible word, almost all empty,
so only the filled slots are stored.
"""

from pathlib import Path

from fastembed import SparseEmbedding, SparseTextEmbedding

from docsgroundtruth.domain import SparseVector

# Not a neural network: just stopword lists and settings, about 10 MB, downloaded once.
_BM25_MODEL = "Qdrant/bm25"


class Bm25Encoder:
    """Implements the `SparseEncoder` interface (see domain.py) with fastembed's BM25."""

    def __init__(self, cache_dir: Path, avg_len: float) -> None:
        """
        Args:
            cache_dir: Folder where fastembed keeps its downloaded files.
            avg_len: Typical chunk length in words, used by BM25's length adjustment.
        """
        self._model = SparseTextEmbedding(
            model_name=_BM25_MODEL, cache_dir=str(cache_dir), avg_len=avg_len
        )

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Word weights for each chunk: mentions with diminishing returns, adjusted for length."""
        return [_to_sparse_vector(embedding) for embedding in self._model.embed(texts)]

    def encode_query(self, text: str) -> SparseVector:
        """The question's words, each with weight 1. Qdrant supplies the rarity weights."""
        return _to_sparse_vector(next(iter(self._model.query_embed(text))))


def _to_sparse_vector(embedding: SparseEmbedding) -> SparseVector:
    """Convert fastembed's result (numpy arrays) into our plain domain type."""
    return SparseVector(
        indices=tuple(int(index) for index in embedding.indices),
        values=tuple(float(value) for value in embedding.values),
    )
