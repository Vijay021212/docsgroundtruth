"""
Application configuration.

Every tunable value lives in this one typed class, instead of being scattered
through the code as magic strings and numbers.

Each value is resolved in this order (first match wins):
    1. A real environment variable, e.g.  $env:DGT_CHUNK_MAX_TOKENS = "500"
    2. A `.env` file in the folder you run the command from
    3. The default written below

Every variable uses the `DGT_` prefix (DocsGroundTruth), so the field
`chunk_max_tokens` is set with the variable `DGT_CHUNK_MAX_TOKENS`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from docsgroundtruth.domain import SearchMode


class Settings(BaseSettings):
    """All application settings, validated by type.

    Pydantic checks types when the settings load, so a typo such as
    DGT_CHUNK_MAX_TOKENS=abc fails immediately with a clear error,
    not halfway through an ingestion run.
    """

    model_config = SettingsConfigDict(env_prefix="DGT_", env_file=".env", extra="ignore")

    # --- Ollama: the local server that runs our models -----------------------
    ollama_host: str = "http://localhost:11434"

    # Model that turns text into vectors: the "R" (retrieval) in RAG.
    embed_model: str = "nomic-embed-text"

    # Model that writes answers: the "G" (generation) in RAG. Used from Phase 4.
    # The full tag is pinned on purpose: a short tag like `qwen3:4b` can be
    # re-pointed to a different model upstream (that's how we got a "thinking" model).
    chat_model: str = "qwen3:4b-instruct-2507-q4_K_M"

    # nomic-embed-text was trained with task prefixes: it expects to be told
    # whether a text is a document or a search query, and it places them better
    # on its "meaning map" when it is. Other embedding models may use different
    # prefixes or none at all (set them to "").
    embed_document_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "

    # Chunks sent to Ollama per request. Bigger = fewer HTTP round trips but more memory.
    embed_batch_size: int = 16

    # --- Chunking ----------------------------------------------------------------
    chunk_max_tokens: int = 800

    # Tokenizer used to COUNT tokens while chunking. It must match the embedding
    # model, otherwise "800 tokens" would be measured with the wrong ruler.
    # This is the Hugging Face copy of the same nomic-embed-text model Ollama runs.
    tokenizer_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # OCR reads text from scanned pages and images. It is accurate but slow on a CPU;
    # turn it off for PDFs whose text you can already select and copy.
    enable_ocr: bool = True

    # --- Retrieval -------------------------------------------------------------------
    # Default search mode: vector, keyword, hybrid or rerank (see SearchMode in domain.py).
    search_mode: SearchMode = SearchMode.RERANK

    # How many candidates each search hands on to fusion and the reranker. More
    # candidates can surface more good chunks, but the reranker must read every one.
    retrieval_candidates: int = 30

    # Reranker model, run by fastembed. Small and fast by default; fastembed also
    # offers bigger, slower ones (e.g. "BAAI/bge-reranker-base") to compare in Phase 5.
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # BM25's "typical chunk length in words", used for its length adjustment.
    # 256 is fastembed's default; measuring our real average is a Phase 5 task.
    bm25_avg_len: float = 256.0

    # Where fastembed keeps downloaded models. Its default is the system temp folder,
    # which Windows disk clean-up can empty (forcing re-downloads), so we use a permanent one.
    model_cache_dir: Path = Path.home() / ".cache" / "fastembed"

    # --- Storage -------------------------------------------------------------------
    raw_data_dir: Path = Path("data/raw")
    qdrant_path: Path = Path("data/index")
    collection_name: str = "docs"

    # --- Logging -------------------------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and reuse them, rather than re-reading `.env` on every call."""
    return Settings()
