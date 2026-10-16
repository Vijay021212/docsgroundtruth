"""
Testing playground: a web page for asking questions and inspecting what retrieval finds.

Start it with:   uv run docsgroundtruth ui

How Streamlit runs this file: it executes the WHOLE script, top to bottom,
every time you interact with the page. Two consequences shape this code:

* The question box sits inside a form, so typing doesn't trigger a run;
  only pressing "Search" does.
* The index summary in the sidebar is cached for a short time, so it isn't
  re-read from disk on every run.

The index is opened only at the moment it's needed and closed right after.
Qdrant's local mode allows one program at a time, so this keeps the index
free for `docsgroundtruth ingest` while the page is open.
"""

import streamlit as st

from docsgroundtruth.config import get_settings
from docsgroundtruth.domain import DocsGroundTruthError, SearchHit
from docsgroundtruth.factory import build_retriever, open_store
from docsgroundtruth.retrieval import RetrievalResult
from docsgroundtruth.stats import SourceStats, summarize

# How long (seconds) the sidebar reuses its index summary before reading the index again.
_SUMMARY_CACHE_SECONDS = 30
# Upper limit for the "number of results" slider.
_MAX_RESULTS = 20


def main() -> None:
    """Draw the whole page. Streamlit calls this on every run (see module docstring)."""
    st.set_page_config(page_title="DocsGroundTruth", page_icon="🔎", layout="wide")
    st.title("DocsGroundTruth playground")
    st.caption("Ask a question and inspect which chunks retrieval finds, and how well they score.")

    _render_sidebar()

    # Widgets inside a form don't trigger a run until the submit button is pressed.
    with st.form("search"):
        question = st.text_input("Question", placeholder="e.g. What is utilization review?")
        top_k = st.slider("Number of results", min_value=1, max_value=_MAX_RESULTS, value=5)
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        _search_and_render(question, top_k)


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
        st.caption(f"Embeddings: {settings.embed_model}")
        st.caption("Search: vector similarity (cosine)")


@st.cache_data(ttl=_SUMMARY_CACHE_SECONDS, show_spinner=False)
def _load_index_summary() -> list[SourceStats]:
    """Per-document statistics, cached for a short time (errors are never cached)."""
    with open_store(get_settings()) as store:
        return summarize(store.iter_chunks())


# --- Search results ----------------------------------------------------------------


def _search_and_render(question: str, top_k: int) -> None:
    """Run retrieval for `question` and show the results, or a clear error."""
    settings = get_settings()
    try:
        with st.spinner("Searching..."), open_store(settings) as store:
            result = build_retriever(settings, store).retrieve(question, top_k=top_k)
    except DocsGroundTruthError as error:
        st.error(str(error))
        return
    _render_results(result)


def _render_results(result: RetrievalResult) -> None:
    """Show the timing summary, then one card per result."""
    if not result.hits:
        st.warning("No results: the index is empty.")
        return
    st.caption(result.summary)
    for rank, hit in enumerate(result.hits, start=1):
        _render_hit(rank, hit)


def _render_hit(rank: int, hit: SearchHit) -> None:
    """One result card: rank, score, source and pages, a preview, and the full chunk on demand."""
    chunk = hit.chunk
    with st.container(border=True):
        st.markdown(
            f"**#{rank}** · score **{hit.score:.3f}** · {chunk.source} · p. {chunk.pages_label}"
        )
        if chunk.headings:
            st.caption(" › ".join(chunk.headings))  # the section path, e.g. Chapter › Section
        # st.text shows text exactly as written. st.markdown would treat characters
        # like * or $ in the document as formatting.
        st.text(chunk.preview())
        with st.expander("Show full chunk"):
            st.text(chunk.text)
            st.caption(f"{chunk.token_count} tokens · chunk #{chunk.index} of this document")


# Streamlit runs this file as the "__main__" module on every run.
if __name__ == "__main__":
    main()
