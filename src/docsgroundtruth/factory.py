"""
Composition root: the ONE place that picks concrete classes and wires them together.

Every other module receives its collaborators ready-made. If we ever swap
Ollama for another provider, this file is where that decision changes.
The CLI uses these builders today, and the web API (Phase 6) will reuse them,
so the wiring is never duplicated.
"""

from docsgroundtruth.config import Settings
from docsgroundtruth.embedding import OllamaEmbedder
from docsgroundtruth.pipeline import IngestionPipeline
from docsgroundtruth.vector_store import QdrantStore


def build_embedder(settings: Settings) -> OllamaEmbedder:
    """Create the embedder described by the settings."""
    return OllamaEmbedder(
        host=settings.ollama_host,
        model=settings.embed_model,
        document_prefix=settings.embed_document_prefix,
        query_prefix=settings.embed_query_prefix,
        batch_size=settings.embed_batch_size,
    )


def open_store(settings: Settings) -> QdrantStore:
    """Open the vector store. Use it in a `with` block so it's always closed."""
    return QdrantStore.open_local(settings.qdrant_path, settings.collection_name)


def build_ingestion_pipeline(settings: Settings, store: QdrantStore) -> IngestionPipeline:
    """Create the full ingestion pipeline on top of an already-open store."""
    # Imported here rather than at the top of the file: Docling takes several
    # seconds to import, and commands like `search` never need it.
    from docsgroundtruth.chunking import DoclingChunker

    chunker = DoclingChunker(
        tokenizer_model=settings.tokenizer_model,
        max_tokens=settings.chunk_max_tokens,
        enable_ocr=settings.enable_ocr,
    )
    return IngestionPipeline(chunker=chunker, embedder=build_embedder(settings), store=store)
