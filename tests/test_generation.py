"""Tests for AnswerGenerator, using a fake chat model instead of Ollama."""

import re
from collections.abc import Iterator

import pytest

from docsgroundtruth.domain import Chunk, DocsGroundTruthError, SearchHit, SearchMode
from docsgroundtruth.generation import NOT_IN_DOCUMENTS, AnswerGenerator, build_prompt
from docsgroundtruth.retrieval import RetrievalResult


class FakeChatModel:
    """Stand-in for Ollama: replies with fixed text and remembers whether it was called."""

    def __init__(self, reply: str = "") -> None:
        self._reply = reply
        self.called = False

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.called = True
        yield from re.findall(r"\S+\s*", self._reply)  # one word at a time, like the real model


def _retrieval(*scores: float, mode: SearchMode = SearchMode.RERANK) -> RetrievalResult:
    """A retrieval result with one hit per score, on pages 1, 2, 3, ..."""
    hits = tuple(
        SearchHit(
            chunk=Chunk(source="a.pdf", index=i, text=f"text {i}", pages=(i + 1,)), score=score
        )
        for i, score in enumerate(scores)
    )
    return RetrievalResult(question="What is X?", mode=mode, hits=hits, timings={})


def _answer(reply: str, *scores: float):
    return AnswerGenerator(FakeChatModel(reply), min_relevance=0.01).answer(_retrieval(*scores))


def test_answer_links_each_citation_to_its_chunk():
    answer = _answer("X is a plan [2]. It pays claims [1].", 0.9, 0.8)

    assert answer.found
    assert [citation.number for citation in answer.citations] == [1, 2]
    assert answer.citations[1].hit.chunk.index == 1  # [2] is the second chunk
    assert answer.citations[1].label == "[2] a.pdf, p. 2"
    assert answer.warnings == ()


def test_several_numbers_in_one_citation_are_understood():
    answer = _answer("X is a plan [1, 2].", 0.9, 0.8)

    assert [citation.number for citation in answer.citations] == [1, 2]


def test_text_is_streamed_piece_by_piece():
    tokens: list[str] = []
    generator = AnswerGenerator(FakeChatModel("X is a plan [1]."), min_relevance=0.01)

    answer = generator.answer(_retrieval(0.9), on_token=tokens.append)

    assert len(tokens) > 1
    assert "".join(tokens) == answer.text


def test_a_citation_to_a_source_that_does_not_exist_gives_a_warning():
    answer = _answer("X is a plan [1]. It pays claims [7].", 0.9)

    assert answer.unknown_citations == (7,)
    assert [citation.number for citation in answer.citations] == [1]
    assert "[7]" in answer.warnings[0]


def test_an_answer_without_citations_gives_a_warning():
    answer = _answer("X is a plan.", 0.9)

    assert answer.found
    assert "no valid citations" in answer.warnings[0]


def test_low_relevance_refuses_without_calling_the_model():
    chat_model = FakeChatModel("This must not be used.")
    tokens: list[str] = []

    answer = AnswerGenerator(chat_model, min_relevance=0.01).answer(
        _retrieval(0.001), on_token=tokens.append
    )

    assert not chat_model.called
    assert not answer.found
    assert answer.text == NOT_IN_DOCUMENTS
    assert tokens == [NOT_IN_DOCUMENTS]  # interfaces still get the text to show


def test_no_results_refuses_without_calling_the_model():
    chat_model = FakeChatModel("This must not be used.")

    answer = AnswerGenerator(chat_model, min_relevance=0.01).answer(_retrieval())

    assert not chat_model.called
    assert not answer.found


def test_the_model_saying_the_answer_is_missing_is_recognised():
    answer = _answer(NOT_IN_DOCUMENTS, 0.9)

    assert not answer.found
    assert answer.warnings == ()  # no "missing citations" warning for a refusal


def test_a_refusal_with_slightly_different_wording_is_recognised():
    answer = _answer("The documents do not contain an answer to this question.", 0.9)

    assert not answer.found


def test_a_refusal_sentence_after_a_real_answer_is_removed():
    answer = _answer(f"X is a plan [1]. {NOT_IN_DOCUMENTS}", 0.9)

    assert answer.found
    assert answer.text == "X is a plan [1]."


def test_an_empty_reply_gives_a_clear_error():
    with pytest.raises(DocsGroundTruthError, match="empty answer"):
        _answer("", 0.9)


def test_results_from_other_search_modes_are_rejected():
    generator = AnswerGenerator(FakeChatModel("X"), min_relevance=0.01)

    # Other modes use different score scales, so min_relevance would mean nothing.
    with pytest.raises(ValueError, match="RERANK"):
        generator.answer(_retrieval(0.9, mode=SearchMode.VECTOR))


def test_prompt_numbers_the_sources_and_includes_the_question():
    retrieval = _retrieval(0.9, 0.8)

    prompt = build_prompt(retrieval.question, retrieval.hits)

    assert "[1] (a.pdf, p. 1)\ntext 0" in prompt
    assert "[2] (a.pdf, p. 2)\ntext 1" in prompt
    assert prompt.endswith("Question: What is X?")
