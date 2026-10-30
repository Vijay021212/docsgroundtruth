"""
Web API, so other programs can use the system.

    uv run docsgroundtruth api      # then open http://127.0.0.1:8000/docs

    GET  /health    is the service up?
    POST /search    the chunks most relevant to a question
    POST /ask       a written answer with page sources

It has no login, so by default it only accepts connections from this computer.
Put real authentication in front of it before exposing it to a network.

This module only translates between web requests and the system's own objects.
The work is done by the two functions passed to `create_app`, so tests can use
fakes without needing Ollama, models or a database.
"""

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from docsgroundtruth.domain import DocsGroundTruthError, SearchHit, SearchMode
from docsgroundtruth.generation import Answer
from docsgroundtruth.retrieval import RetrievalResult

# Limits on what a caller can send, so one request can't ask for unreasonable work.
_MAX_QUESTION_CHARS = 2000
_MAX_RESULTS = 20


class _Question(BaseModel):
    """Shared by both requests. Stripping spaces first means a blank question is rejected."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)


class AskRequest(_Question):
    pass


class SearchRequest(_Question):
    top_k: int = Field(default=5, ge=1, le=_MAX_RESULTS)
    mode: SearchMode = SearchMode.RERANK


class ChunkResponse(BaseModel):
    source: str
    pages: list[int]
    headings: list[str]
    text: str


class HitResponse(BaseModel):
    chunk: ChunkResponse
    score: float
    ranks: dict[str, int]


class SearchResponse(BaseModel):
    question: str
    mode: SearchMode
    score_name: str  # what `score` means in this mode, e.g. "relevance"
    hits: list[HitResponse]
    timings: dict[str, float]


class CitationResponse(BaseModel):
    number: int
    chunk: ChunkResponse


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool  # False when the documents don't contain the answer
    citations: list[CitationResponse]
    warnings: list[str]


def _chunk_response(hit: SearchHit) -> ChunkResponse:
    chunk = hit.chunk
    return ChunkResponse(
        source=chunk.source,
        pages=list(chunk.pages),
        headings=list(chunk.headings),
        text=chunk.text,
    )


def create_app(
    retrieve: Callable[[str, int, SearchMode], RetrievalResult],
    write_answer: Callable[[RetrievalResult], Answer],
    answer_chunks: int,
) -> FastAPI:
    """Build the web app.

    Args:
        retrieve: Finds chunks, given the question, the number of results and the mode.
        write_answer: Writes an answer from a RERANK retrieval result.
        answer_chunks: How many chunks each answer is written from.
    """
    app = FastAPI(
        title="DocsGroundTruth",
        description="Answers questions from your own documents, with page sources.",
    )

    @app.exception_handler(DocsGroundTruthError)
    def explain_problem(request: Request, error: DocsGroundTruthError) -> JSONResponse:
        """Expected problems (Ollama down, index busy, nothing indexed) mean "try again later"."""
        return JSONResponse(status_code=503, content={"detail": str(error)})

    # These are plain `def` functions, so FastAPI runs them in worker threads. Slow
    # model calls then don't stop the server from accepting other requests.
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search")
    def search(request: SearchRequest) -> SearchResponse:
        result = retrieve(request.question, request.top_k, request.mode)
        return SearchResponse(
            question=result.question,
            mode=result.mode,
            score_name=result.mode.score_name,
            hits=[
                HitResponse(chunk=_chunk_response(hit), score=hit.score, ranks=hit.ranks)
                for hit in result.hits
            ],
            timings=result.timings,
        )

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        retrieval = retrieve(request.question, answer_chunks, SearchMode.RERANK)
        answer = write_answer(retrieval)
        return AskResponse(
            question=retrieval.question,
            answer=answer.text,
            found=answer.found,
            citations=[
                CitationResponse(number=citation.number, chunk=_chunk_response(citation.hit))
                for citation in answer.citations
            ],
            warnings=list(answer.warnings),
        )

    return app
