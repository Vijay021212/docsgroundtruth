"""
Evaluation: give the system a test paper and score how it does.

Each question in the test paper states what should happen: the right page is
found, or the system refuses. Running them all turns "it looks good" into numbers,
so we can tell whether a change really makes the system better.

What the code can score by itself:

* Search      - was an expected page in the top results? Compared across all four modes.
* Refusals    - did it refuse the questions the documents can't answer, and only those?
* Key words   - does the answer contain the words a correct answer should?
* Citations   - did it cite an expected page, and how many sentences have a citation?

What it can't score: whether an answer is really TRUE to its source. Key words only
suggest that. So each run also writes a review sheet for a person to check.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from docsgroundtruth.domain import DocsGroundTruthError, SearchHit, SearchMode
from docsgroundtruth.generation import CITATION, Answer, AnswerGenerator
from docsgroundtruth.retrieval import RetrievalResult, Retriever

# One sentence, plus any citations right after its full stop: "A claim. [1]" stays together.
# Approximate: abbreviations such as "U.S." count as sentence ends.
_SENTENCE = re.compile(r"[^.!?]+[.!?]*(?:\s*\[[\d,\s]+\])*")


class QuestionType(StrEnum):
    """What should happen for a test question."""

    IN_DOCUMENT = "in_document"  # the answer is in the documents
    NOT_IN_DOCUMENT = "not_in_document"  # related topic, but not answered: should refuse
    OFF_TOPIC = "off_topic"  # unrelated: should refuse


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """One entry in the test paper."""

    id: str
    type: QuestionType
    question: str
    pages: tuple[int, ...] = ()  # where the answer is (in_document only)
    must_include: tuple[str, ...] = ()  # words a correct answer should contain (in_document only)


@dataclass(frozen=True, slots=True)
class Score:
    """A result such as 17 out of 20."""

    passed: int
    total: int

    def __str__(self) -> str:
        if not self.total:
            return "n/a"
        return f"{self.passed}/{self.total} ({self.passed / self.total:.0%})"


@dataclass(frozen=True, slots=True)
class ModeScore:
    """How well one search mode finds the expected pages."""

    mode: SearchMode
    hits: Score  # questions with an expected page in the top results
    mean_reciprocal_rank: float  # 1 = always first, 0.5 = always second, 0 = never found


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """What the system did for one question, and the checks on it."""

    question: EvalQuestion
    retrieval: RetrievalResult
    answer: Answer

    @property
    def page_found(self) -> bool:
        return rank_of_expected_page(self.retrieval.hits, self.question.pages) is not None

    @property
    def key_words_found(self) -> bool:
        text = " ".join(self.answer.text.lower().split())
        return all(word.lower() in text for word in self.question.must_include)

    @property
    def cited_expected_page(self) -> bool:
        return any(
            set(citation.hit.chunk.pages) & set(self.question.pages)
            for citation in self.answer.citations
        )


@dataclass(frozen=True, slots=True)
class EvalReport:
    """All the results of one test run."""

    results: tuple[QuestionResult, ...]
    mode_scores: tuple[ModeScore, ...]

    def of_type(self, question_type: QuestionType) -> list[QuestionResult]:
        return [result for result in self.results if result.question.type is question_type]

    def refusals(self, question_type: QuestionType) -> Score:
        """How many questions of this type the system refused."""
        results = self.of_type(question_type)
        return Score(sum(not result.answer.found for result in results), len(results))

    def _in_document_score(self, check: Callable[[QuestionResult], bool]) -> Score:
        results = self.of_type(QuestionType.IN_DOCUMENT)
        return Score(sum(check(result) for result in results), len(results))

    @property
    def answered(self) -> Score:
        """In-document questions it answered instead of wrongly refusing."""
        return self._in_document_score(lambda result: result.answer.found)

    @property
    def key_words(self) -> Score:
        return self._in_document_score(lambda result: result.key_words_found)

    @property
    def cited_expected_pages(self) -> Score:
        return self._in_document_score(lambda result: result.cited_expected_page)

    @property
    def sentences_with_citations(self) -> Score:
        """Across all in-document answers, how many sentences have a citation."""
        shares = [
            citation_share(result.answer.text)
            for result in self.of_type(QuestionType.IN_DOCUMENT)
            if result.answer.found
        ]
        return Score(sum(share.passed for share in shares), sum(share.total for share in shares))


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read and check a test paper file."""
    try:
        entries = yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]
        questions = [_to_question(entry) for entry in entries]
    except FileNotFoundError as error:
        raise DocsGroundTruthError(f"Test paper not found: {path}") from error
    except (yaml.YAMLError, KeyError, TypeError, ValueError) as error:
        raise DocsGroundTruthError(f"Test paper {path} is not valid: {error}") from error
    if not questions:
        raise DocsGroundTruthError(f"Test paper {path} has no questions.")
    return questions


def _to_question(entry: dict) -> EvalQuestion:
    """Convert one entry of the file, checking in-document questions say what to look for."""
    question = EvalQuestion(
        id=entry["id"],
        type=QuestionType(entry["type"]),
        question=entry["question"],
        pages=tuple(entry.get("pages", ())),
        must_include=tuple(str(word) for word in entry.get("must_include", ())),
    )
    if question.type is QuestionType.IN_DOCUMENT and not (question.pages and question.must_include):
        raise ValueError(f"{question.id} needs both `pages` and `must_include`")
    return question


def rank_of_expected_page(hits: tuple[SearchHit, ...], pages: tuple[int, ...]) -> int | None:
    """Position (1 = first) of the first hit on an expected page, or None if there's none."""
    for rank, hit in enumerate(hits, start=1):
        if set(hit.chunk.pages) & set(pages):
            return rank
    return None


def citation_share(text: str) -> Score:
    """How many sentences of a text have a citation like [1]."""
    # Pieces without any letters are leftovers, such as a citation alone on a line.
    sentences = [part for part in _SENTENCE.findall(text) if re.search(r"[a-zA-Z]", part)]
    return Score(sum(bool(CITATION.search(part)) for part in sentences), len(sentences))


class Evaluator:
    """Runs a test paper through the real search and answer steps."""

    def __init__(self, retriever: Retriever, generator: AnswerGenerator, top_k: int) -> None:
        """
        Args:
            top_k: How many chunks to retrieve, the same number real answers use.
        """
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    def run(
        self,
        questions: list[EvalQuestion],
        on_result: Callable[[int, QuestionResult], None] = lambda position, result: None,
    ) -> EvalReport:
        """Ask every question, reporting each result as it's ready, then compare search modes.

        Args:
            on_result: Called with the position (1, 2, ...) and result of each question,
                because a full run takes minutes and progress should be visible.
        """
        results = []
        for position, question in enumerate(questions, start=1):
            result = self._ask(question)
            results.append(result)
            on_result(position, result)
        return EvalReport(results=tuple(results), mode_scores=self._compare_modes(questions))

    def _ask(self, question: EvalQuestion) -> QuestionResult:
        retrieval = self._retriever.retrieve(
            question.question, top_k=self._top_k, mode=SearchMode.RERANK
        )
        return QuestionResult(question, retrieval, self._generator.answer(retrieval))

    def _compare_modes(self, questions: list[EvalQuestion]) -> tuple[ModeScore, ...]:
        """Score each search mode on the in-document questions. No answers are written."""
        answerable = [q for q in questions if q.type is QuestionType.IN_DOCUMENT]
        return tuple(self._score_mode(mode, answerable) for mode in SearchMode)

    def _score_mode(self, mode: SearchMode, questions: list[EvalQuestion]) -> ModeScore:
        ranks = [
            rank_of_expected_page(
                self._retriever.retrieve(question.question, top_k=self._top_k, mode=mode).hits,
                question.pages,
            )
            for question in questions
        ]
        found = [rank for rank in ranks if rank is not None]
        return ModeScore(
            mode=mode,
            hits=Score(len(found), len(ranks)),
            mean_reciprocal_rank=sum(1 / rank for rank in found) / len(ranks) if ranks else 0.0,
        )
