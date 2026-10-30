"""Tests for choosing where a parsed document is saved."""

from docsgroundtruth.parse_cache import parsed_document_path


def test_the_same_file_and_setting_give_the_same_path(tmp_path):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"version 1")

    assert parsed_document_path(tmp_path, source, True) == parsed_document_path(
        tmp_path, source, True
    )


def test_changing_the_file_gives_a_new_path(tmp_path):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"version 1")
    before = parsed_document_path(tmp_path, source, True)

    source.write_bytes(b"version 2")

    assert parsed_document_path(tmp_path, source, True) != before


def test_the_ocr_setting_gives_a_different_path(tmp_path):
    source = tmp_path / "a.pdf"
    source.write_bytes(b"version 1")

    assert parsed_document_path(tmp_path, source, True) != parsed_document_path(
        tmp_path, source, False
    )
