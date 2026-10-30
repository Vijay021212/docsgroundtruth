"""Tests for QdrantStore, run against a real Qdrant instance held in memory."""

import pytest
from qdrant_client import QdrantClient, models

from docsgroundtruth.domain import DocsGroundTruthError, SparseVector
from docsgroundtruth.vector_store import QdrantStore


def test_vector_search_returns_the_closest_meaning_first(
    store, embedder, index_chunks, make_chunks
):
    texts = ["prior authorization", "claims processing", "member eligibility"]
    chunks = make_chunks("a.pdf", texts)
    index_chunks(chunks)

    hits = store.vector_search(embedder.embed_query("claims processing"), top_k=3)

    assert hits[0].chunk == chunks[1]
    assert hits[0].score == pytest.approx(1.0)  # identical vectors -> cosine similarity of 1
    assert len(hits) == 3


def test_keyword_search_finds_the_exact_term(store, keyword_encoder, index_chunks, make_chunks):
    texts = ["the HIN links providers", "a data warehouse stores data", "claims paid monthly"]
    chunks = make_chunks("a.pdf", texts)
    index_chunks(chunks)

    hits = store.keyword_search(keyword_encoder.encode_query("HIN"), top_k=3)

    assert hits[0].chunk == chunks[0]


def test_keyword_search_with_no_usable_words_returns_nothing(store, index_chunks, make_chunks):
    index_chunks(make_chunks("a.pdf", ["some text"]))

    assert store.keyword_search(SparseVector(), top_k=3) == []


def test_reingesting_a_source_replaces_only_that_sources_chunks(store, index_chunks, make_chunks):
    index_chunks(make_chunks("a.pdf", ["one", "two", "three"]))
    index_chunks(make_chunks("b.pdf", ["other document"]))

    # a.pdf changed and now produces a single chunk.
    index_chunks(make_chunks("a.pdf", ["one (revised)"]))

    # 1 new chunk for a.pdf + 1 untouched chunk for b.pdf. No stale "two"/"three" left behind.
    assert store.count() == 2


def test_collection_rejects_a_different_vector_size(store):
    store.ensure_collection(8)

    with pytest.raises(DocsGroundTruthError, match="8-dimensional"):
        store.ensure_collection(16)


def test_an_index_from_before_keyword_search_is_explained():
    # Build a collection the way Phase 2 did: one unnamed vector, no keyword vectors.
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "docs", vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE)
    )

    with QdrantStore(client, "docs") as old_store:
        with pytest.raises(DocsGroundTruthError, match="older version"):
            old_store.ensure_collection(8)
        with pytest.raises(DocsGroundTruthError, match="older version"):
            old_store.vector_search([0.1] * 8, top_k=1)


def test_iter_chunks_returns_every_stored_chunk(store, index_chunks, make_chunks):
    chunks = make_chunks("a.pdf", [f"chunk {i}" for i in range(5)])
    index_chunks(chunks)

    # A page size smaller than the number of chunks forces several pages,
    # so this also tests that the paging loop doesn't skip or repeat chunks.
    stored = list(store.iter_chunks(page_size=2))

    assert sorted(stored, key=lambda chunk: chunk.index) == chunks


def test_reading_before_any_ingestion_explains_what_to_do(store, embedder, keyword_encoder):
    with pytest.raises(DocsGroundTruthError, match="ingest"):
        store.vector_search(embedder.embed_query("anything"), top_k=1)
    with pytest.raises(DocsGroundTruthError, match="ingest"):
        store.keyword_search(keyword_encoder.encode_query("anything"), top_k=1)
    with pytest.raises(DocsGroundTruthError, match="ingest"):
        list(store.iter_chunks())  # list() is needed: a generator only runs when read


def test_opening_an_index_that_is_in_use_explains_the_problem(tmp_path):
    # The first store holds the folder's lock, like an ingest run or the playground would.
    with (
        QdrantStore.open_local(tmp_path, "docs"),
        pytest.raises(DocsGroundTruthError, match="in use by another program"),
    ):
        QdrantStore.open_local(tmp_path, "docs")
