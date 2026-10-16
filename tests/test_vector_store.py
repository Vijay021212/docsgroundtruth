"""Tests for QdrantStore, run against a real Qdrant instance held in memory."""

import pytest

from docsgroundtruth.domain import DocsGroundTruthError


def _index(store, embedder, chunks):
    """Embed `chunks` and store them, the same way the pipeline does."""
    store.ensure_collection(embedder.dimension)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store.replace_source(chunks[0].source, chunks, vectors)


def test_search_returns_most_similar_chunk_first(store, embedder, make_chunks):
    texts = ["prior authorization", "claims processing", "member eligibility"]
    chunks = make_chunks("a.pdf", texts)
    _index(store, embedder, chunks)

    hits = store.search(embedder.embed_query("claims processing"), top_k=3)

    assert hits[0].chunk == chunks[1]
    assert hits[0].score == pytest.approx(1.0)  # identical vectors -> cosine similarity of 1
    assert len(hits) == 3


def test_reingesting_a_source_replaces_only_that_sources_chunks(store, embedder, make_chunks):
    _index(store, embedder, make_chunks("a.pdf", ["one", "two", "three"]))
    _index(store, embedder, make_chunks("b.pdf", ["other document"]))

    # a.pdf changed and now produces a single chunk.
    _index(store, embedder, make_chunks("a.pdf", ["one (revised)"]))

    # 1 new chunk for a.pdf + 1 untouched chunk for b.pdf. No stale "two"/"three" left behind.
    assert store.count() == 2


def test_collection_rejects_a_different_vector_size(store):
    store.ensure_collection(8)

    with pytest.raises(DocsGroundTruthError, match="8-dimensional"):
        store.ensure_collection(16)


def test_iter_chunks_returns_every_stored_chunk(store, embedder, make_chunks):
    chunks = make_chunks("a.pdf", [f"chunk {i}" for i in range(5)])
    _index(store, embedder, chunks)

    # A page size smaller than the number of chunks forces several pages,
    # so this also tests that the paging loop doesn't skip or repeat chunks.
    stored = list(store.iter_chunks(page_size=2))

    assert sorted(stored, key=lambda chunk: chunk.index) == chunks


def test_reading_before_any_ingestion_explains_what_to_do(store, embedder):
    with pytest.raises(DocsGroundTruthError, match="ingest"):
        store.search(embedder.embed_query("anything"))
    with pytest.raises(DocsGroundTruthError, match="ingest"):
        list(store.iter_chunks())  # list() is needed: a generator only runs when read
