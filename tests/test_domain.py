"""Tests for the core domain types."""

import uuid

from docsgroundtruth.domain import Chunk


def test_chunk_id_is_deterministic_and_unique_per_position():
    first = Chunk(source="a.pdf", index=0, text="hello")
    same_position = Chunk(source="a.pdf", index=0, text="different text")
    next_position = Chunk(source="a.pdf", index=1, text="hello")

    # The ID depends only on (source, index), so re-ingesting overwrites instead of duplicating.
    assert first.id == same_position.id
    assert first.id != next_position.id
    uuid.UUID(first.id)  # raises if it isn't a valid UUID, which Qdrant requires


def test_payload_round_trip_preserves_every_field():
    chunk = Chunk(
        source="a.pdf",
        index=3,
        text="Claims\nA claim is a request for payment.",
        pages=(4, 5),
        headings=("Chapter 2", "Claims"),
        token_count=12,
    )

    assert Chunk.from_payload(chunk.to_payload()) == chunk


def test_pages_label_lists_the_pages_or_a_question_mark():
    assert Chunk(source="a.pdf", index=0, text="x", pages=(308, 309)).pages_label == "308, 309"
    assert Chunk(source="notes.md", index=0, text="x").pages_label == "?"


def test_preview_is_a_single_short_line():
    chunk = Chunk(source="a.pdf", index=0, text="Claims\n\nA claim is a request for payment. " * 20)

    preview = chunk.preview(max_chars=60)

    assert "\n" not in preview
    assert len(preview) <= 60
    assert preview.endswith(" ...")
