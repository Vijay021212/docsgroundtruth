"""Tests for Retriever and Reciprocal Rank Fusion, using fakes and an in-memory store."""

import pytest

from docsgroundtruth.domain import Chunk, DocsGroundTruthError, SearchHit, SearchMode
from docsgroundtruth.retrieval import RRF_K, Retriever, reciprocal_rank_fusion


@pytest.fixture
def retriever(store, embedder, keyword_encoder, reranker) -> Retriever:
    # A small candidate pool keeps the tests easy to reason about.
    return Retriever(embedder, keyword_encoder, reranker, store, candidates=10)


def _hit(chunk: Chunk) -> SearchHit:
    """A search hit for fusion tests. RRF ignores raw scores, so 0 is fine."""
    return SearchHit(chunk=chunk, score=0.0)


# --- Reciprocal Rank Fusion ------------------------------------------------------------


def test_rrf_rewards_chunks_found_high_by_both_searches(make_chunks):
    a, b, c = make_chunks("a.pdf", ["A", "B", "C"])

    fused = reciprocal_rank_fusion(
        {"vector": [_hit(a), _hit(b), _hit(c)], "keyword": [_hit(c), _hit(a)]}, limit=3
    )

    # A: 1/61 + 1/62 (positions 1 and 2) beats C: 1/63 + 1/61, which beats B: 1/62 only.
    assert [hit.chunk for hit in fused] == [a, c, b]
    assert fused[0].ranks == {"vector": 1, "keyword": 2}
    assert fused[2].score == pytest.approx(1 / (RRF_K + 2))


def test_rrf_returns_at_most_limit_results(make_chunks):
    chunks = make_chunks("a.pdf", ["A", "B", "C"])

    fused = reciprocal_rank_fusion({"vector": [_hit(chunk) for chunk in chunks]}, limit=2)

    assert len(fused) == 2


# --- The four search modes -------------------------------------------------------------


def test_vector_mode_finds_the_closest_meaning(retriever, index_chunks, make_chunks):
    texts = ["prior authorization", "claims processing", "member eligibility"]
    chunks = make_chunks("a.pdf", texts)
    index_chunks(chunks)

    result = retriever.retrieve("  claims processing  ", top_k=2, mode=SearchMode.VECTOR)

    assert result.question == "claims processing"  # surrounding spaces are removed
    assert result.hits[0].chunk == chunks[1]
    assert len(result.hits) == 2
    assert list(result.timings) == ["question embedding", "vector search"]
    assert result.summary.startswith("2 results in ")


def test_summary_says_result_for_a_single_hit(retriever, index_chunks, make_chunks):
    index_chunks(make_chunks("a.pdf", ["claims processing", "member eligibility"]))

    result = retriever.retrieve("claims processing", top_k=1, mode=SearchMode.VECTOR)

    assert result.summary.startswith("1 result in ")


def test_keyword_mode_matches_exact_words(retriever, index_chunks, make_chunks):
    chunks = make_chunks("a.pdf", ["the HIN links providers", "data warehouse", "claims paid"])
    index_chunks(chunks)

    result = retriever.retrieve("HIN", mode=SearchMode.KEYWORD)

    assert result.hits[0].chunk == chunks[0]
    assert list(result.timings) == ["keyword search"]


def test_hybrid_mode_records_where_each_search_ranked_a_chunk(
    retriever, index_chunks, make_chunks
):
    texts = ["claims processing", "claims paid monthly", "member eligibility"]
    chunks = make_chunks("a.pdf", texts)
    index_chunks(chunks)

    result = retriever.retrieve("claims processing", top_k=3, mode=SearchMode.HYBRID)

    top = result.hits[0]
    assert top.chunk == chunks[0]  # first in BOTH searches
    assert top.ranks == {"vector": 1, "keyword": 1}
    assert top.score == pytest.approx(2 / (RRF_K + 1))
    assert list(result.timings) == [
        "question embedding",
        "vector search",
        "keyword search",
        "fusion",
    ]


def test_rerank_mode_orders_candidates_by_relevance(retriever, index_chunks, make_chunks):
    texts = ["claims paid monthly", "how claims processing works", "member eligibility"]
    chunks = make_chunks("a.pdf", texts)
    index_chunks(chunks)

    result = retriever.retrieve("claims processing", top_k=2, mode=SearchMode.RERANK)

    # FakeReranker: relevance = share of the question's words found in the text.
    assert [hit.chunk for hit in result.hits] == [chunks[1], chunks[0]]
    assert [hit.score for hit in result.hits] == pytest.approx([1.0, 0.5])
    assert result.hits[0].ranks  # where the searches placed it is kept after reranking
    assert "reranking" in result.timings


# --- Input validation ------------------------------------------------------------------


# parametrize runs the same test once per value, here: an empty and a blank question.
@pytest.mark.parametrize("question", ["", "   "])
def test_blank_question_is_rejected(retriever, question):
    with pytest.raises(DocsGroundTruthError, match="enter a question"):
        retriever.retrieve(question)


def test_number_of_results_must_be_at_least_one(retriever):
    with pytest.raises(DocsGroundTruthError, match="at least 1"):
        retriever.retrieve("anything", top_k=0)
