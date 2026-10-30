"""
Document parsing and chunking, powered by Docling.

Two steps happen here:

1. PARSE - Docling reads the file and understands its layout: headings,
   paragraphs, tables, reading order and page numbers (plus OCR for scanned
   pages). The result is a structured `DoclingDocument`, not a flat blob of text.

2. CHUNK - Docling's HybridChunker cuts that structured document into pieces:
   * first along the document's own structure (sections, tables, lists),
   * then it splits anything still bigger than `max_tokens`,
   * then it merges small neighbouring pieces that share the same heading.
   So chunks follow the author's sections instead of being cut blindly every
   N characters, often mid-sentence.

Each Docling chunk is converted into our own `Chunk` type, so nothing outside
this module needs to know that Docling exists.
"""

import logging
from pathlib import Path

from docling.chunking import DocChunk, HybridChunker
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument, ImageRefMode

from docsgroundtruth.domain import Chunk
from docsgroundtruth.parse_cache import parsed_document_path
from docsgroundtruth.timing import Stopwatch

logger = logging.getLogger(__name__)


class DoclingChunker:
    """Implements the `Chunker` interface (see domain.py) using Docling."""

    # Formats Docling can parse that make sense as knowledge sources.
    supported_suffixes = frozenset({".pdf", ".docx", ".pptx", ".html", ".md"})

    def __init__(
        self, tokenizer_model: str, max_tokens: int, cache_dir: Path, enable_ocr: bool = True
    ) -> None:
        """
        Args:
            tokenizer_model: Hugging Face tokenizer used to measure chunk sizes.
            max_tokens: Upper limit on the size of one chunk, in tokens.
            cache_dir: Folder where parsed documents are saved and reused.
            enable_ocr: Read text from scanned pages and images (slow on CPU).
        """
        self._cache_dir = cache_dir
        self._enable_ocr = enable_ocr

        # The converter does the parsing. Only the PDF pipeline is customised
        # (OCR on/off); every other format keeps Docling's defaults.
        pdf_options = PdfPipelineOptions(do_ocr=enable_ocr)
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
        )

        # The tokenizer is the "ruler" the chunker measures with. The first run
        # downloads the tokenizer files (about 1 MB) from Hugging Face; later runs use the cache.
        self._tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=tokenizer_model, max_tokens=max_tokens
        )
        # merge_peers=True: glue small neighbouring chunks under the same heading
        # together, so we don't end up with chunks that are only one line long.
        self._chunker = HybridChunker(tokenizer=self._tokenizer, merge_peers=True)

    def chunk_file(self, path: Path) -> list[Chunk]:
        """Parse `path` and return its chunks in reading order, logging how long each step took."""
        with Stopwatch() as watch:
            document = self._parse(path)
        logger.info("Parsed %d pages in %.1fs", document.num_pages(), watch.seconds)

        with Stopwatch() as watch:
            chunks = [
                self._to_chunk(doc_chunk, source=path.name, index=index)
                for index, doc_chunk in enumerate(self._chunker.chunk(dl_doc=document))
            ]
        logger.info("Chunked into %d chunks in %.1fs", len(chunks), watch.seconds)
        return chunks

    def _parse(self, path: Path) -> DoclingDocument:
        """Run Docling's layout analysis on the file (the slow step), or reuse a saved result."""
        saved = parsed_document_path(self._cache_dir, path, self._enable_ocr)
        if saved.exists():
            logger.info("Using the saved parse of %s", path.name)
            return DoclingDocument.load_from_json(saved)

        logger.info("Parsing %s (OCR %s)...", path.name, "on" if self._enable_ocr else "off")
        result = self._converter.convert(path)

        # When SOME pages fail, Docling doesn't raise an error: it returns what it
        # could convert and marks the result PARTIAL_SUCCESS. We report that loudly,
        # because silently missing pages would mean silently incomplete answers.
        if result.status == ConversionStatus.PARTIAL_SUCCESS:
            details = "; ".join(error.error_message for error in result.errors) or "no details"
            logger.warning(
                "Only part of %s could be converted, so some pages may be missing. Details: %s",
                path.name,
                details,
            )
        else:
            # Only complete results are saved, so failed pages get another try next time.
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            result.document.save_as_json(saved, image_mode=ImageRefMode.PLACEHOLDER)
        return result.document

    def _to_chunk(self, doc_chunk: DocChunk, source: str, index: int) -> Chunk:
        """Convert a Docling chunk into our own `Chunk` type."""
        # contextualize() puts the chunk's section headings in front of its text.
        # "Prior Authorization\nRequests are processed within 48 hours" retrieves
        # much better than the bare sentence, because the heading carries meaning
        # the sentence alone doesn't have.
        text = self._chunker.contextualize(chunk=doc_chunk)

        # Each element in the chunk (paragraph, table, list item) records where
        # it came from: its "provenance". We collect the distinct page numbers,
        # which later become citations such as "(p. 4-5)".
        pages = sorted({prov.page_no for item in doc_chunk.meta.doc_items for prov in item.prov})

        return Chunk(
            source=source,
            index=index,
            text=text,
            pages=tuple(pages),
            headings=tuple(doc_chunk.meta.headings or ()),
            token_count=self._tokenizer.count_tokens(text=text),
        )
