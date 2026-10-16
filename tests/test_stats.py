"""Tests for the index statistics calculations (pure functions, no database needed)."""

from docsgroundtruth.domain import Chunk
from docsgroundtruth.stats import SizeBucket, summarize


def _chunk(source: str, index: int, tokens: int, pages: tuple[int, ...]) -> Chunk:
    """A chunk where only the fields used by the statistics matter."""
    return Chunk(
        source=source, index=index, text=f"chunk {index}", pages=pages, token_count=tokens
    )


def test_reports_chunk_sizes_per_document_sorted_by_name():
    chunks = [
        _chunk("b.pdf", 0, 120, (1,)),
        _chunk("a.pdf", 0, 50, (1,)),
        _chunk("a.pdf", 1, 250, (2,)),
        _chunk("a.pdf", 2, 800, (3,)),
    ]

    a_stats, b_stats = summarize(chunks)

    assert a_stats.source == "a.pdf"
    assert (a_stats.chunk_count, a_stats.min_tokens, a_stats.max_tokens) == (3, 50, 800)
    assert a_stats.median_tokens == 250
    assert (b_stats.source, b_stats.chunk_count) == ("b.pdf", 1)


def test_histogram_counts_chunks_per_size_range():
    chunks = [_chunk("a.pdf", i, tokens, (1,)) for i, tokens in enumerate([100, 101, 800])]

    (stats,) = summarize(chunks, bucket_size=400)

    # A chunk of exactly 800 tokens belongs in "401-800", not in a bar of its own.
    assert stats.size_histogram == (
        SizeBucket(low=1, high=400, count=2),
        SizeBucket(low=401, high=800, count=1),
    )


def test_pages_without_text_are_grouped_into_ranges():
    chunks = [_chunk("a.pdf", 0, 10, (3, 4)), _chunk("a.pdf", 1, 10, (8,))]

    (stats,) = summarize(chunks)

    assert stats.last_page == 8
    assert stats.pages_without_text == ((1, 2), (5, 7))


def test_documents_without_page_numbers_report_no_pages():
    (stats,) = summarize([_chunk("notes.md", 0, 10, ())])

    assert stats.last_page is None
    assert stats.pages_without_text == ()


def test_no_chunks_gives_an_empty_report():
    assert summarize([]) == []
