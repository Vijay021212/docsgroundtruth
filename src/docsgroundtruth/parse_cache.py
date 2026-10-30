"""
Where a parsed copy of a document is saved.

Parsing is the slow step of ingestion (about 7 minutes for our 472-page PDF), so
each result is saved and reused. That makes it quick to try different chunk sizes.
"""

import hashlib
from pathlib import Path


def parsed_document_path(cache_dir: Path, source: Path, enable_ocr: bool) -> Path:
    """The file for this exact document and OCR setting.

    The name includes a fingerprint of the file's contents, so changing the
    document automatically means parsing it again instead of using an old copy.
    """
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    ocr = "ocr" if enable_ocr else "no-ocr"
    return cache_dir / f"{source.stem}-{fingerprint}-{ocr}.json"
