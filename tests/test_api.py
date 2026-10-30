"""Tests for the web API, using fake search and answer functions."""

import pytest
from fastapi.testclient import TestClient

from docsgroundtruth.api import create_app
from docsgroundtruth.domain import Chunk, DocsGroundTruthError, SearchHit, SearchMode
from docsgroundtruth.generation import Answer, Citation
from docsgroundtruth.retrieval import RetrievalResult

_HIT = SearchHit(
    chunk=Chunk(source="a.pdf", index=0, text="A HIN is a network.", pages=(3,)),
    score=0.9,
    ranks={"vector": 1},
)


def _retrieve(question: str, top_k: int, mode: SearchMode) -> RetrievalResult:
    return RetrievalResult(question=question, mode=mode, hits=(_HIT,), timings={"search": 0.1})


def _write_answer(retrieval: RetrievalResult) -> Answer:
    return Answer(text="A network [1].", found=True, citations=(Citation(1, _HIT),))


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(_retrieve, _write_answer, answer_chunks=5))


def test_health_reports_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_the_answer_and_its_sources(client):
    response = client.post("/ask", json={"question": "  What is a HIN?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is a HIN?"  # surrounding spaces are removed
    assert body["answer"] == "A network [1]."
    assert body["found"] is True
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["chunk"]["pages"] == [3]


def test_search_returns_scored_chunks(client):
    response = client.post("/search", json={"question": "HIN", "top_k": 1, "mode": "keyword"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "keyword"
    assert body["score_name"] == "BM25 score"
    assert body["hits"][0]["chunk"]["text"] == "A HIN is a network."


# Each entry is a request the API must refuse before doing any work.
@pytest.mark.parametrize(
    "request_body",
    [
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "HIN", "top_k": 0},
        {"question": "HIN", "top_k": 21},
        {"question": "HIN", "mode": "magic"},
    ],
)
def test_invalid_search_requests_are_rejected(client, request_body):
    assert client.post("/search", json=request_body).status_code == 422


def test_expected_problems_are_reported_clearly():
    def index_busy(question: str, top_k: int, mode: SearchMode) -> RetrievalResult:
        raise DocsGroundTruthError("The index is in use.")

    client = TestClient(create_app(index_busy, _write_answer, answer_chunks=5))

    response = client.post("/ask", json={"question": "HIN"})

    assert response.status_code == 503
    assert response.json() == {"detail": "The index is in use."}
