"""Tests for Retriever, using the fake embedder and an in-memory store."""

import pytest

from docsgroundtruth.domain import DocsGroundTruthError
from docsgroundtruth.retrieval import Retriever


@pytest.fixture
def retriever(store, embedder) -> Retriever:
    return Retriever(embedder, store)


def test_returns_the_best_matches_first_with_step_timings(retriever, index_chunks, make_chunks):
    texts = ["prior authorization", "claims processing", "member eligibility"]
    chunks = make_chunks("a.pdf", texts)
    index_chunks(chunks)

    result = retriever.retrieve("  claims processing  ", top_k=2)

    assert result.question == "claims processing"  # surrounding spaces are removed
    assert result.hits[0].chunk == chunks[1]
    assert len(result.hits) == 2
    assert list(result.timings) == ["question embedding", "vector search"]
    assert result.summary.startswith("2 results in ")


# parametrize runs the same test once per value, here: an empty and a blank question.
@pytest.mark.parametrize("question", ["", "   "])
def test_blank_question_is_rejected(retriever, question):
    with pytest.raises(DocsGroundTruthError, match="enter a question"):
        retriever.retrieve(question)


def test_number_of_results_must_be_at_least_one(retriever):
    with pytest.raises(DocsGroundTruthError, match="at least 1"):
        retriever.retrieve("anything", top_k=0)
