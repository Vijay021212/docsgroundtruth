"""
Turning evaluation results into text: progress lines, a summary, and a review sheet.

The review sheet lets a person judge what the code can't: whether each answer is
really true to its sources. It's saved as Markdown, with the sources next to the answer.
"""

from statistics import median

from docsgroundtruth.evaluation import EvalReport, QuestionResult, QuestionType


def _yes_no(value: bool) -> str:
    return "yes" if value else "NO"


def format_progress(result: QuestionResult) -> str:
    """One line about a finished question."""
    question = result.question
    if question.type is QuestionType.IN_DOCUMENT:
        checks = (
            f"page found: {_yes_no(result.page_found)}, "
            f"answered: {_yes_no(result.answer.found)}, "
            f"key words: {_yes_no(result.key_words_found)}"
        )
    else:
        checks = f"refused: {_yes_no(not result.answer.found)}"
    return f"{question.id} ({result.answer.seconds:.0f}s) {checks}"


def format_summary(report: EvalReport) -> str:
    """The scores of a run, as plain text."""
    lines = ["Search: expected page in the top results", ""]
    for score in report.mode_scores:
        lines.append(
            f"  {score.mode.label:<28}{str(score.hits):<14}"
            f"mean reciprocal rank {score.mean_reciprocal_rank:.2f}"
        )

    lines += [
        "",
        "Answers to questions the documents can answer",
        "",
        f"  Answered, not wrongly refused   {report.answered}",
        f"  Contain the key words           {report.key_words}",
        f"  Cite an expected page           {report.cited_expected_pages}",
        f"  Sentences with a citation       {report.sentences_with_citations}",
        "",
        "Refusals",
        "",
        f"  Healthcare, not in the document {report.refusals(QuestionType.NOT_IN_DOCUMENT)}",
        f"  Off topic                       {report.refusals(QuestionType.OFF_TOPIC)}",
    ]

    # Only answers the model wrote: instant refusals would make writing look faster.
    written = [result.answer.seconds for result in report.results if result.answer.seconds]
    if written:
        lines += ["", f"Typical time to write an answer: {median(written):.1f}s"]
    return "\n".join(lines)


def format_review_sheet(report: EvalReport) -> str:
    """A Markdown sheet for a person to judge each answer against its sources."""
    sections = [
        "# Evaluation review sheet",
        "```\n" + format_summary(report) + "\n```",
        "For each answer: is every claim supported by the sources shown? Mark yes or no.",
    ]
    sections += [_review_entry(result) for result in report.results]
    return "\n\n".join(sections) + "\n"


def _review_entry(result: QuestionResult) -> str:
    """The question, the answer and its sources, for one question."""
    question = result.question
    hits = result.retrieval.hits
    expected = ", ".join(map(str, question.pages)) or "(should refuse)"
    found_pages = "; ".join(hit.chunk.pages_label for hit in hits) or "none"
    best_score = f"{hits[0].score:.3f}" if hits else "none"
    parts = [
        f"## {question.id}: {question.question}",
        f"- Type: {question.type}\n"
        f"- Expected pages: {expected}\n"
        f"- Pages retrieved: {found_pages}\n"
        f"- Best relevance score: {best_score}",
        f"**Answer:**\n\n{result.answer.text}",
    ]
    parts += [f"**Warning:** {warning}" for warning in result.answer.warnings]
    parts += [
        f"**Source {citation.label}**\n\n> " + citation.hit.chunk.text.replace("\n", "\n> ")
        for citation in result.answer.citations
    ]
    parts.append("Correct? [ ] yes  [ ] no")
    return "\n\n".join(parts)
