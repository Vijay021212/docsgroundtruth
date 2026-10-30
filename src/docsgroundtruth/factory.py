"""
Composition root: the ONE place that picks concrete classes and wires them together.

Every other module receives its collaborators ready-made. If we ever swap
Ollama or fastembed for another provider, this file is where that decision
changes. The CLI and the playground use these builders today, and the web API
(Phase 6) will reuse them, so the wiring is never duplicated.

Docling, fastembed and FastAPI are imported inside the builders that need them,
not at the top of this file: they take time to import, and commands like `stats`
never use them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docsgroundtruth.chat import OllamaChatModel
from docsgroundtruth.config import Settings
from docsgroundtruth.domain import Embedder, Reranker, SearchMode, SparseEncoder
from docsgroundtruth.embedding import OllamaEmbedder
from docsgroundtruth.evaluation import Evaluator
from docsgroundtruth.generation import AnswerGenerator
from docsgroundtruth.pipeline import IngestionPipeline
from docsgroundtruth.retrieval import RetrievalResult, Retriever
from docsgroundtruth.vector_store import QdrantStore

if TYPE_CHECKING:  # only for the type hint: commands other than `api` don't need FastAPI
    from fastapi import FastAPI


@dataclass(frozen=True, slots=True)
class SearchModels:
    """The models a Retriever needs.

    Building them takes a moment (loading files, checking downloads), so a
    long-running interface like the playground builds them once and reuses them.
    """

    embedder: Embedder
    keyword_encoder: SparseEncoder
    reranker: Reranker


def build_embedder(settings: Settings) -> OllamaEmbedder:
    """Create the embedder (meaning vectors) described by the settings."""
    return OllamaEmbedder(
        host=settings.ollama_host,
        model=settings.embed_model,
        document_prefix=settings.embed_document_prefix,
        query_prefix=settings.embed_query_prefix,
        batch_size=settings.embed_batch_size,
    )


def build_keyword_encoder(settings: Settings) -> SparseEncoder:
    """Create the BM25 keyword encoder."""
    from docsgroundtruth.bm25 import Bm25Encoder

    return Bm25Encoder(cache_dir=settings.model_cache_dir, avg_len=settings.bm25_avg_len)


def build_reranker(settings: Settings) -> Reranker:
    """Create the reranker. Its model is downloaded the first time."""
    from docsgroundtruth.reranking import FastEmbedReranker

    return FastEmbedReranker(model=settings.rerank_model, cache_dir=settings.model_cache_dir)


def build_search_models(settings: Settings) -> SearchModels:
    """Create every model retrieval needs."""
    return SearchModels(
        embedder=build_embedder(settings),
        keyword_encoder=build_keyword_encoder(settings),
        reranker=build_reranker(settings),
    )


def open_store(settings: Settings) -> QdrantStore:
    """Open the vector store. Use it in a `with` block so it's always closed."""
    return QdrantStore.open_local(settings.qdrant_path, settings.collection_name)


def build_retriever(
    settings: Settings, store: QdrantStore, models: SearchModels | None = None
) -> Retriever:
    """Create the retriever on top of an already-open store.

    Pass `models` to reuse models built earlier; otherwise they're built here.
    """
    models = models or build_search_models(settings)
    return Retriever(
        embedder=models.embedder,
        keyword_encoder=models.keyword_encoder,
        reranker=models.reranker,
        store=store,
        candidates=settings.retrieval_candidates,
    )


def build_answer_generator(settings: Settings) -> AnswerGenerator:
    """Create the generator that writes answers with the local chat model."""
    chat_model = OllamaChatModel(
        host=settings.ollama_host,
        model=settings.chat_model,
        context_tokens=settings.chat_context_tokens,
        max_answer_tokens=settings.chat_max_answer_tokens,
    )
    return AnswerGenerator(chat_model=chat_model, min_relevance=settings.min_relevance)


def build_evaluator(settings: Settings, store: QdrantStore) -> Evaluator:
    """Create the evaluator, using the same search and answer steps as real questions."""
    return Evaluator(
        retriever=build_retriever(settings, store),
        generator=build_answer_generator(settings),
        top_k=settings.answer_chunks,
    )


def build_api_app(settings: Settings) -> FastAPI:
    """Create the web API, with its search models built once and reused for every request."""
    from docsgroundtruth.api import create_app

    models = build_search_models(settings)
    generator = build_answer_generator(settings)
    # Qdrant's local mode allows one open connection at a time, so requests take turns
    # searching. The slow answer writing happens outside the lock, so it doesn't block them.
    index_lock = threading.Lock()

    def retrieve(question: str, top_k: int, mode: SearchMode) -> RetrievalResult:
        with index_lock, open_store(settings) as store:
            retriever = build_retriever(settings, store, models)
            return retriever.retrieve(question, top_k=top_k, mode=mode)

    return create_app(retrieve, generator.answer, settings.answer_chunks)


def build_ingestion_pipeline(settings: Settings, store: QdrantStore) -> IngestionPipeline:
    """Create the full ingestion pipeline on top of an already-open store."""
    from docsgroundtruth.chunking import DoclingChunker

    chunker = DoclingChunker(
        tokenizer_model=settings.tokenizer_model,
        max_tokens=settings.chunk_max_tokens,
        cache_dir=settings.parsed_docs_dir,
        enable_ocr=settings.enable_ocr,
    )
    return IngestionPipeline(
        chunker=chunker,
        embedder=build_embedder(settings),
        keyword_encoder=build_keyword_encoder(settings),
        store=store,
    )
