"""Tests for the evaluation scoring, using fakes and an in-memory store."""

from pathlib import Path

import pytest

from docsgroundtruth.domain import Chunk, DocsGroundTruthError, SearchHit, SearchMode
from docsgroundtruth.evaluation import (
    EvalQuestion,
    Evaluator,
    QuestionType,
    Score,
    citation_share,
    load_questions,
    rank_of_expected_page,
)
from docsgroundtruth.evaluation_report import format_review_sheet, format_summary
from docsgroundtruth.generation import AnswerGenerator
from docsgroundtruth.retrieval import Retriever


class FakeChatModel:
    """Stand-in for Ollama: cites source 1 and repeats the words of the question."""

    def stream(self, system, user):
        yield f"{user.rsplit('Question: ', 1)[1]} [1]."


def _hit(*pages: int) -> SearchHit:
    return SearchHit(chunk=Chunk(source="a.pdf", index=0, text="x", pages=pages), score=0.9)


def _write_paper(tmp_path, text: str):
    path = tmp_path / "paper.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- Small building blocks -------------------------------------------------------------


def test_rank_of_expected_page_finds_the_first_match():
    hits = (_hit(3), _hit(7, 8), _hit(8))

    assert rank_of_expected_page(hits, (8,)) == 2
    assert rank_of_expected_page(hits, (99,)) is None


def test_citation_share_counts_sentences_with_a_citation():
    assert citation_share("One claim [1]. Another claim. A third. [2]") == Score(2, 3)


def test_score_reads_well_and_handles_nothing_to_score():
    assert str(Score(17, 20)) == "17/20 (85%)"
    assert str(Score(0, 0)) == "n/a"


# --- Reading the test paper ------------------------------------------------------------


def test_load_questions_reads_a_test_paper(tmp_path):
    path = _write_paper(
        tmp_path,
        """
questions:
  - {id: q1, type: in_document, question: Why, pages: [3], must_include: ["85"]}
  - {id: o1, type: off_topic, question: Pizza}
""",
    )

    questions = load_questions(path)

    assert questions[0].pages == (3,)
    assert questions[0].must_include == ("85",)
    assert questions[1].type is QuestionType.OFF_TOPIC


def test_the_real_test_paper_is_valid():
    questions = load_questions(Path(__file__).parents[1] / "eval" / "ahm250_questions.yaml")

    assert len(questions) == 30
    assert len({question.id for question in questions}) == 30  # no repeated ids


@pytest.mark.parametrize(
    "text",
    [
        "questions: []",
        "questions:\n  - {id: q1, type: unknown, question: Why}",
        "questions:\n  - {id: q1, type: in_document, question: Why}",  # no pages or key words
        "not: [the: right, shape",
    ],
)
def test_invalid_test_papers_are_explained(tmp_path, text):
    with pytest.raises(DocsGroundTruthError, match="Test paper"):
        load_questions(_write_paper(tmp_path, text))


def test_a_missing_test_paper_is_explained(tmp_path):
    with pytest.raises(DocsGroundTruthError, match="not found"):
        load_questions(tmp_path / "missing.yaml")


# --- A full run ------------------------------------------------------------------------


@pytest.fixture
def report(store, embedder, keyword_encoder, reranker, index_chunks):
    index_chunks(
        [
            Chunk(source="a.pdf", index=0, text="claims processing rules", pages=(5,)),
            Chunk(source="a.pdf", index=1, text="member eligibility rules", pages=(9,)),
        ]
    )
    retriever = Retriever(embedder, keyword_encoder, reranker, store, candidates=10)
    generator = AnswerGenerator(FakeChatModel(), min_relevance=0.01)
    questions = [
        EvalQuestion(
            id="q1",
            type=QuestionType.IN_DOCUMENT,
            question="claims processing rules",
            pages=(5,),
            must_include=("claims",),
        ),
        # FakeReranker scores this 0: no shared words with any chunk, so it's refused.
        EvalQuestion(id="o1", type=QuestionType.OFF_TOPIC, question="pizza"),
    ]
    return Evaluator(retriever, generator, top_k=2).run(questions)


def test_a_run_scores_answers_and_refusals(report):
    assert str(report.answered) == "1/1 (100%)"
    assert str(report.key_words) == "1/1 (100%)"
    assert str(report.cited_expected_pages) == "1/1 (100%)"
    assert str(report.sentences_with_citations) == "1/1 (100%)"
    assert str(report.refusals(QuestionType.OFF_TOPIC)) == "1/1 (100%)"


def test_a_run_compares_every_search_mode(report):
    assert [score.mode for score in report.mode_scores] == list(SearchMode)
    reranked = report.mode_scores[-1]
    assert reranked.hits == Score(1, 1)
    assert reranked.mean_reciprocal_rank == pytest.approx(1.0)


def test_progress_is_reported_for_each_question(
    store, embedder, keyword_encoder, reranker, index_chunks, make_chunks
):
    index_chunks(make_chunks("a.pdf", ["claims"]))
    retriever = Retriever(embedder, keyword_encoder, reranker, store, candidates=10)
    generator = AnswerGenerator(FakeChatModel(), min_relevance=0.01)
    seen = []

    Evaluator(retriever, generator, top_k=1).run(
        [EvalQuestion(id="o1", type=QuestionType.OFF_TOPIC, question="claims")],
        on_result=lambda position, result: seen.append((position, result.question.id)),
    )

    assert seen == [(1, "o1")]


def test_the_summary_and_review_sheet_show_the_results(report):
    assert "Hybrid + reranker" in format_summary(report)

    sheet = format_review_sheet(report)
    assert "## q1: claims processing rules" in sheet
    assert "Correct? [ ] yes  [ ] no" in sheet
    assert "> claims processing rules" in sheet  # the source is next to the answer
