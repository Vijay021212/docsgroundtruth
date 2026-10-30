"""
Command-line interface.

    uv run docsgroundtruth ingest [PATH]            # index a file or folder (default: data/raw)
    uv run docsgroundtruth ask "your question"      # get a written answer with page sources
    uv run docsgroundtruth search "your question"   # show the most relevant chunks
    uv run docsgroundtruth search "..." -m hybrid   # ... using a specific search mode
    uv run docsgroundtruth stats                    # report card for the indexed chunks
    uv run docsgroundtruth evaluate                 # run the test paper and report the scores
    uv run docsgroundtruth ui                       # open the testing playground in your browser
    uv run docsgroundtruth api                      # start the web API on this computer
    uv run docsgroundtruth --help                   # list commands and options
"""

import argparse
import contextlib
import io
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from docsgroundtruth.config import Settings, get_settings
from docsgroundtruth.domain import DocsGroundTruthError, SearchHit, SearchMode
from docsgroundtruth.evaluation import QuestionResult, load_questions
from docsgroundtruth.evaluation_report import (
    format_progress,
    format_review_sheet,
    format_summary,
)
from docsgroundtruth.factory import (
    build_answer_generator,
    build_api_app,
    build_evaluator,
    build_ingestion_pipeline,
    build_retriever,
    open_store,
)
from docsgroundtruth.stats import SizeBucket, SourceStats, summarize
from docsgroundtruth.ui import PLAYGROUND_SCRIPT

logger = logging.getLogger(__name__)

# Length, in characters, of the longest bar in the stats histogram.
_BAR_WIDTH = 30


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code: 0 = success, 1 = failure."""
    settings = get_settings()
    args = _build_parser(settings).parse_args(argv)
    _configure_logging(settings.log_level)
    _make_output_safe()

    try:
        # Each sub-command stored its handler function in `args.handler` (see _build_parser).
        args.handler(args, settings)
    except DocsGroundTruthError as error:
        # Expected problems (Ollama down, nothing indexed yet, ...) get a clear
        # one-line message instead of a long traceback.
        logger.error("%s", error)
        return 1
    return 0


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    """Define the commands and their arguments."""
    parser = argparse.ArgumentParser(
        prog="docsgroundtruth",
        description="Answer questions from your own documents, with citations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Parse, chunk, embed and index documents.")
    ingest.add_argument(
        "path",
        nargs="?",  # optional: falls back to the default below
        type=Path,
        default=settings.raw_data_dir,
        help="File or folder to ingest (default: %(default)s).",
    )
    ingest.set_defaults(handler=_run_ingest)

    ask = commands.add_parser("ask", help="Get a written answer with page sources.")
    ask.add_argument("question", help="The question to answer, in quotes.")
    ask.set_defaults(handler=_run_ask)

    search = commands.add_parser("search", help="Show the chunks most relevant to a question.")
    search.add_argument("query", help="The question to search for, in quotes.")
    search.add_argument(
        "-k", "--top-k", type=int, default=5, help="How many results to show (default: 5)."
    )
    search.add_argument(
        "-m",
        "--mode",
        type=SearchMode,  # turns the typed text, e.g. "hybrid", into a SearchMode
        choices=list(SearchMode),
        default=settings.search_mode,
        help="How to search (default: %(default)s).",
    )
    search.set_defaults(handler=_run_search)

    stats = commands.add_parser("stats", help="Show chunk counts, sizes and page coverage.")
    stats.set_defaults(handler=_run_stats)

    evaluate = commands.add_parser("evaluate", help="Run a test paper and report the scores.")
    evaluate.add_argument(
        "questions",
        nargs="?",  # optional: falls back to the default below
        type=Path,
        default=settings.eval_questions,
        help="Test paper file (default: %(default)s).",
    )
    evaluate.set_defaults(handler=_run_evaluate)

    ui = commands.add_parser("ui", help="Open the testing playground in your browser.")
    ui.set_defaults(handler=_run_ui)

    api = commands.add_parser("api", help="Start the web API on this computer.")
    api.set_defaults(handler=_run_api)

    return parser


# --- Command handlers ------------------------------------------------------------


def _run_ingest(args: argparse.Namespace, settings: Settings) -> None:
    """`ingest` command: run the ingestion pipeline and print a summary."""
    with open_store(settings) as store:
        reports = build_ingestion_pipeline(settings, store).ingest_path(args.path)
        total = store.count()

    print("\nIngestion summary:")
    for report in reports:
        print(f"  {report.source}: {report.chunk_count} chunks in {report.seconds:.1f}s")
    print(f"The index now holds {total} chunks.")


def _run_ask(args: argparse.Namespace, settings: Settings) -> None:
    """`ask` command: retrieve the best chunks, then print the answer as it's written."""
    with open_store(settings) as store:
        retrieval = build_retriever(settings, store).retrieve(
            args.question, top_k=settings.answer_chunks, mode=SearchMode.RERANK
        )
    # The index is closed by now, so the slow writing step doesn't keep it locked.

    print("Answer:\n")
    answer = build_answer_generator(settings).answer(
        retrieval, on_token=lambda token: print(token, end="", flush=True)
    )
    print()

    for warning in answer.warnings:
        print(f"\nWarning: {warning}")
    if answer.citations:
        print("\nSources:")
        for citation in answer.citations:
            print(f"  {citation.label}")
    print(f"\n{retrieval.summary}\nAnswer written in {answer.seconds:.1f}s")


def _run_search(args: argparse.Namespace, settings: Settings) -> None:
    """`search` command: retrieve the chunks most relevant to the question and print them."""
    with open_store(settings) as store:
        retriever = build_retriever(settings, store)
        result = retriever.retrieve(args.query, top_k=args.top_k, mode=args.mode)

    print(f"Search mode: {result.mode.label}")
    if not result.hits:
        print("No results.")
        return
    for rank, hit in enumerate(result.hits, start=1):
        print(_format_hit(rank, hit, result.mode))
    print(f"\n{result.summary}")


def _run_stats(args: argparse.Namespace, settings: Settings) -> None:
    """`stats` command: print a report card for every ingested document."""
    with open_store(settings) as store:
        # iter_chunks() reads lazily, page by page, so summarize() must run
        # while the store is still open, i.e. inside this `with` block.
        report = summarize(store.iter_chunks())

    if not report:
        print("The index is empty.")
        return
    for source_stats in report:
        print(_format_source_stats(source_stats))
    total = sum(source_stats.chunk_count for source_stats in report)
    print(f"\nTotal: {len(report)} document(s), {total} chunks")


def _run_evaluate(args: argparse.Namespace, settings: Settings) -> None:
    """`evaluate` command: run the test paper, print the scores, save a review sheet."""
    questions = load_questions(args.questions)
    print(f"Running {len(questions)} questions. This can take 10-15 minutes.\n")

    def show_progress(position: int, result: QuestionResult) -> None:
        print(f"[{position}/{len(questions)}] {format_progress(result)}", flush=True)

    # The index stays open for the whole run, so the playground can't search until it ends.
    with open_store(settings) as store:
        report = build_evaluator(settings, store).run(questions, on_result=show_progress)

    print(f"\n{format_summary(report)}")

    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    sheet = settings.eval_results_dir / f"review_{datetime.now():%Y%m%d_%H%M}.md"
    sheet.write_text(format_review_sheet(report), encoding="utf-8")
    print(f"\nReview sheet saved to {sheet}")


def _run_ui(args: argparse.Namespace, settings: Settings) -> None:
    """`ui` command: start the Streamlit playground. Stop it with Ctrl+C."""
    command = [
        # The current Python (sys.executable) guarantees the project's .venv is used.
        sys.executable, "-m", "streamlit", "run", str(PLAYGROUND_SCRIPT),
        # Only accept connections from this computer. The page has no login, so
        # without this anyone on the same network could read the indexed documents.
        "--server.address=localhost",
        # Skip Streamlit's first-run question asking for your email address.
        "--server.showEmailPrompt=false",
        # Don't send usage statistics to Streamlit: this is a private, local tool.
        "--browser.gatherUsageStats=false",
    ]
    # Ctrl+C is the normal way to stop the playground, so exit quietly instead
    # of printing a KeyboardInterrupt traceback.
    with contextlib.suppress(KeyboardInterrupt):
        subprocess.run(command, check=False)


def _run_api(args: argparse.Namespace, settings: Settings) -> None:
    """`api` command: start the web API. Stop it with Ctrl+C."""
    import uvicorn  # imported here because only this command needs it

    uvicorn.run(build_api_app(settings), host=settings.api_host, port=settings.api_port)


# --- Output formatting -----------------------------------------------------------


def _format_hit(rank: int, hit: SearchHit, mode: SearchMode) -> str:
    """Render one search result as a short, readable block of text."""
    chunk = hit.chunk
    header = (
        f"#{rank}  {mode.score_name}={hit.score:.3f}  {chunk.source}  (p. {chunk.pages_label})"
    )
    if hit.ranks:  # hybrid modes: show where each search had placed this chunk
        header += f"  [{hit.ranks_label}]"
    return f"\n{header}\n    {chunk.preview()}"


def _format_source_stats(stats: SourceStats) -> str:
    """Render one document's statistics as an indented block of text."""
    lines = [
        f"\n{stats.source}",
        f"  Chunks:              {stats.chunk_count}",
        f"  Chunk size (tokens): smallest {stats.min_tokens} | "
        f"median {stats.median_tokens:.0f} | largest {stats.max_tokens}",
        "  Size distribution (tokens: chunks):",
        *_format_histogram(stats.size_histogram),
    ]
    if stats.last_page is None:
        lines.append("  Pages:               not available for this file type")
    else:
        lines.append(f"  Last page with text: {stats.last_page}")
        lines.append(f"  Pages with no text:  {_format_page_ranges(stats.pages_without_text)}")
    return "\n".join(lines)


def _format_histogram(buckets: tuple[SizeBucket, ...]) -> list[str]:
    """One text bar per size range, scaled so the fullest range gets the longest bar."""
    fullest = max(bucket.count for bucket in buckets)
    label_width = len(f"{buckets[-1].low}-{buckets[-1].high}")  # widest label, for alignment

    lines = []
    for bucket in buckets:
        # Every non-empty range gets at least one block, so small counts stay visible.
        length = max(1, round(bucket.count / fullest * _BAR_WIDTH)) if bucket.count else 0
        label = f"{bucket.low}-{bucket.high}"
        bar = "█" * length
        lines.append(f"    {label:>{label_width}}  {bar:<{_BAR_WIDTH}}  {bucket.count}")
    return lines


def _format_page_ranges(ranges: tuple[tuple[int, int], ...]) -> str:
    """Turn page runs into text: ((1, 2), (156, 156)) -> "1-2, 156"."""
    if not ranges:
        return "none"
    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in ranges)


# --- Setup -------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    """Show our own log messages at `level`, but only warnings from third-party libraries.

    Docling and Hugging Face log a lot at INFO level. This keeps the output readable.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("docsgroundtruth").setLevel(level.upper())


def _make_output_safe() -> None:
    """Never crash because of a character the output can't display.

    PDF text and the histogram's "█" bars can contain characters some outputs
    can't encode, for example when output is redirected to a file on Windows.
    Those characters are printed as "?" instead of raising UnicodeEncodeError.
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")
