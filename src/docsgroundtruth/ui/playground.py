"""
Testing playground: a web page for asking questions and inspecting what retrieval finds.

Start it with:   uv run docsgroundtruth ui

"Get answer" writes an answer with page sources. "Search only" shows the chunks
instead: pick one of the four search modes (vector, keyword, hybrid, hybrid +
reranker), or turn on "Compare" to see all four side by side.

How Streamlit runs this file: it executes the WHOLE script, top to bottom,
every time you interact with the page. That shapes the code:

* The question box sits inside a form, so typing doesn't trigger a run;
  only pressing one of its buttons does.
* The search models (embedder, BM25 encoder, reranker) are built once per
  session and cached, so they aren't reloaded for every search.
* The index summary in the sidebar is cached for a short time, so it isn't
  re-read from disk on every run.

The index itself is opened only while searching and closed right after. Qdrant's
local mode allows one program at a time, so this keeps the index free for
`docsgroundtruth ingest` while the page is open.
"""

import streamlit as st

from docsgroundtruth.config import get_settings
from docsgroundtruth.domain import DocsGroundTruthError, SearchHit, SearchMode
from docsgroundtruth.factory import (
    SearchModels,
    build_answer_generator,
    build_retriever,
    build_search_models,
    open_store,
)
from docsgroundtruth.retrieval import RetrievalResult
from docsgroundtruth.stats import SourceStats, summarize

# How long (seconds) the sidebar reuses its index summary before reading the index again.
_SUMMARY_CACHE_SECONDS = 30
# Upper limit for the "number of results" slider.
_MAX_RESULTS = 20
# Preview length in the Compare view, where four columns share the page width.
_COMPACT_PREVIEW_CHARS = 120


def main() -> None:
    """Draw the whole page. Streamlit calls this on every run (see module docstring)."""
    settings = get_settings()
    st.set_page_config(page_title="DocsGroundTruth", page_icon="🔎", layout="wide")
    st.title("DocsGroundTruth playground")
    st.caption("Ask a question, then get an answer with sources or inspect the chunks found.")

    _render_sidebar()

    # Widgets inside a form don't trigger a run until the submit button is pressed.
    with st.form("search"):
        question = st.text_input("Question", placeholder="e.g. What is a HIN?")
        ask = st.form_submit_button("Get answer", type="primary")

        st.divider()
        st.caption("These settings are for Search only.")
        modes = list(SearchMode)
        mode = st.radio(
            "Search mode",
            options=modes,
            index=modes.index(settings.search_mode),
            format_func=lambda option: option.label,  # show "Hybrid + reranker", not "rerank"
            horizontal=True,
        )
        top_k = st.slider("Number of results", min_value=1, max_value=_MAX_RESULTS, value=5)
        compare = st.toggle("Compare all four modes side by side (ignores the mode above)")
        search = st.form_submit_button("Search only")

    if ask:
        _answer_and_render(question)
    elif search and compare:
        _compare_and_render(question, top_k)
    elif search:
        _search_and_render(question, top_k, mode)


# --- Sidebar ---------------------------------------------------------------------


def _render_sidebar() -> None:
    """Show what's in the index and which models are in use."""
    settings = get_settings()
    with st.sidebar:
        st.subheader("Index")
        try:
            summary = _load_index_summary()
        except DocsGroundTruthError as error:
            st.info(str(error))  # e.g. nothing ingested yet, or index busy
        else:
            st.metric("Documents", len(summary))
            st.metric("Chunks", f"{sum(stats.chunk_count for stats in summary):,}")
            for stats in summary:
                st.caption(f"{stats.source}: {stats.chunk_count:,} chunks")

        st.subheader("Models")
        st.caption(f"Meaning (embeddings): {settings.embed_model}")
        st.caption("Keywords: BM25")
        st.caption(f"Reranker: {settings.rerank_model}")
        st.caption(f"Candidates per search: {settings.retrieval_candidates}")
        st.caption(f"Answers: {settings.chat_model}")


@st.cache_data(ttl=_SUMMARY_CACHE_SECONDS, show_spinner=False)
def _load_index_summary() -> list[SourceStats]:
    """Per-document statistics, cached for a short time (errors are never cached)."""
    with open_store(get_settings()) as store:
        return summarize(store.iter_chunks())


@st.cache_resource(show_spinner="Loading search models (first time only)...")
def _search_models() -> SearchModels:
    """Build the search models once per playground session; every search reuses them."""
    return build_search_models(get_settings())


# --- Answering -----------------------------------------------------------------------


def _answer_and_render(question: str) -> None:
    """Find the best chunks, then show the answer word by word, with warnings and sources."""
    settings = get_settings()
    results = _run_searches(question, settings.answer_chunks, [SearchMode.RERANK])
    if not results:
        return
    retrieval = results[0]

    st.subheader("Answer")
    placeholder = st.empty()  # a spot on the page we can keep rewriting as words arrive
    written = ""

    def show_token(token: str) -> None:
        nonlocal written
        written += token
        placeholder.text(written)  # st.text, so characters like * and $ show as written

    try:
        with st.spinner("Writing the answer (this can take a little while on a CPU)..."):
            answer = build_answer_generator(settings).answer(retrieval, on_token=show_token)
    except DocsGroundTruthError as error:
        st.error(str(error))
        return

    placeholder.text(answer.text)  # the final, cleaned-up text replaces what was streamed
    for warning in answer.warnings:
        st.warning(warning)
    st.caption(f"{retrieval.summary}. Answer written in {answer.seconds:.1f}s.")

    if answer.citations:
        st.subheader("Sources")
        for citation in answer.citations:
            _render_hit(citation.number, citation.hit, retrieval.mode, compact=False)


# --- Searching -----------------------------------------------------------------------


def _search_and_render(question: str, top_k: int, mode: SearchMode) -> None:
    """Search with one mode and show its results."""
    results = _run_searches(question, top_k, [mode])
    if results:
        _render_results(results[0])


def _compare_and_render(question: str, top_k: int) -> None:
    """Search with all four modes and show the results in four columns."""
    results = _run_searches(question, top_k, list(SearchMode))
    if not results:
        return
    for column, result in zip(st.columns(len(results)), results, strict=True):
        with column:
            st.subheader(result.mode.label)
            _render_results(result, compact=True)


def _run_searches(question: str, top_k: int, modes: list[SearchMode]) -> list[RetrievalResult]:
    """Run retrieval once per mode, with the index open only while searching.

    Problems (empty question, nothing indexed, index busy, Ollama down) are
    shown on the page, and an empty list is returned.
    """
    settings = get_settings()
    try:
        with st.spinner("Searching..."), open_store(settings) as store:
            retriever = build_retriever(settings, store, _search_models())
            return [retriever.retrieve(question, top_k=top_k, mode=mode) for mode in modes]
    except DocsGroundTruthError as error:
        st.error(str(error))
        return []


# --- Results ---------------------------------------------------------------------------


def _render_results(result: RetrievalResult, compact: bool = False) -> None:
    """Show the timing summary, then one card per result."""
    if not result.hits:
        # e.g. keyword search when the question has only filler words
        st.warning("No results.")
        return
    st.caption(result.summary)
    for rank, hit in enumerate(result.hits, start=1):
        _render_hit(rank, hit, result.mode, compact)


def _render_hit(rank: int, hit: SearchHit, mode: SearchMode, compact: bool) -> None:
    """One result card: rank, score, pages, where each search placed it, and the text."""
    chunk = hit.chunk
    with st.container(border=True):
        # The file name is left out in the narrow Compare columns.
        source = "" if compact else f" · {chunk.source}"
        st.markdown(
            f"**#{rank}** · {mode.score_name} **{hit.score:.3f}** · p. {chunk.pages_label}{source}"
        )
        if hit.ranks:  # hybrid modes: e.g. "vector #3 · keyword #1"
            st.caption(f"Found by: {hit.ranks_label}")
        if chunk.headings:
            st.caption(" › ".join(chunk.headings))  # the section path, e.g. Chapter › Section
        # st.text shows text exactly as written. st.markdown would treat characters
        # like * or $ in the document as formatting.
        st.text(chunk.preview(_COMPACT_PREVIEW_CHARS) if compact else chunk.preview())
        with st.expander("Show full chunk"):
            st.text(chunk.text)
            st.caption(f"{chunk.token_count} tokens · chunk #{chunk.index} of this document")


# Streamlit runs this file as the "__main__" module on every run.
if __name__ == "__main__":
    main()
