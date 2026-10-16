"""
Index statistics: a "report card" for the chunks stored in the vector store.

It answers questions like:
  * How many chunks did each document produce?
  * How big are the chunks? Are most far below the 800-token limit, or cut at it?
  * Which pages produced no text? Covers and blank pages are normal;
    a long run of missing pages means something went wrong during parsing.

This module only CALCULATES: no database access, no printing. The CLI passes
chunks in and prints the result, and the web API (Phase 6) can reuse it the same way.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

from docsgroundtruth.domain import Chunk

# Width of each bar in the size histogram, in tokens: 1-100, 101-200, 201-300, ...
DEFAULT_BUCKET_SIZE = 100


@dataclass(frozen=True, slots=True)
class SizeBucket:
    """One histogram bar: how many chunks have between `low` and `high` tokens (both included)."""

    low: int
    high: int
    count: int


@dataclass(frozen=True, slots=True)
class SourceStats:
    """Statistics for one ingested document."""

    source: str
    chunk_count: int
    min_tokens: int
    median_tokens: float
    max_tokens: int
    size_histogram: tuple[SizeBucket, ...]
    # Highest page number that produced text. None for formats without pages (e.g. Markdown).
    last_page: int | None
    # Pages without any text, grouped into (first, last) runs: ((1, 2), (156, 156)).
    pages_without_text: tuple[tuple[int, int], ...]


def summarize(
    chunks: Iterable[Chunk], bucket_size: int = DEFAULT_BUCKET_SIZE
) -> list[SourceStats]:
    """Group chunks by source document and compute statistics for each, sorted by file name."""
    chunks_by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk.source].append(chunk)

    return [
        _summarize_source(source, source_chunks, bucket_size)
        for source, source_chunks in sorted(chunks_by_source.items())
    ]


def _summarize_source(source: str, chunks: list[Chunk], bucket_size: int) -> SourceStats:
    """Compute the statistics for the chunks of a single document."""
    sizes = [chunk.token_count for chunk in chunks]
    pages_with_text = {page for chunk in chunks for page in chunk.pages}
    last_page = max(pages_with_text) if pages_with_text else None

    # Pages are numbered from 1. Any page up to the last one that no chunk
    # mentions produced no text.
    missing_pages = (
        sorted(set(range(1, last_page + 1)) - pages_with_text) if last_page is not None else []
    )

    return SourceStats(
        source=source,
        chunk_count=len(chunks),
        min_tokens=min(sizes),
        median_tokens=median(sizes),
        max_tokens=max(sizes),
        size_histogram=_histogram(sizes, bucket_size),
        last_page=last_page,
        pages_without_text=_to_ranges(missing_pages),
    )


def _histogram(sizes: list[int], bucket_size: int) -> tuple[SizeBucket, ...]:
    """Count chunks per size range (1-100, 101-200, ...), up to the largest chunk.

    Ranges include their upper bound, so a chunk of exactly 800 tokens is
    counted in "701-800" with its neighbours instead of alone in an "800-899" bar.
    Empty ranges are kept, so the shape of the distribution stays honest.
    """
    counts = [0] * (_bucket_index(max(sizes), bucket_size) + 1)
    for size in sizes:
        counts[_bucket_index(size, bucket_size)] += 1

    return tuple(
        SizeBucket(low=index * bucket_size + 1, high=(index + 1) * bucket_size, count=count)
        for index, count in enumerate(counts)
    )


def _bucket_index(size: int, bucket_size: int) -> int:
    """Which bar a chunk belongs to: 0 for 1-100 tokens, 1 for 101-200, and so on."""
    return max(size - 1, 0) // bucket_size


def _to_ranges(numbers: list[int]) -> tuple[tuple[int, int], ...]:
    """Collapse sorted numbers into consecutive runs: [1, 2, 3, 7] -> ((1, 3), (7, 7))."""
    ranges: list[tuple[int, int]] = []
    for number in numbers:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], number)  # continues the current run
        else:
            ranges.append((number, number))  # starts a new run
    return tuple(ranges)
