"""
Text -> vector embeddings, computed by a model running in the local Ollama server.

This is the same `/api/embed` call you made by hand with Invoke-RestMethod in
Phase 0, now wrapped in a class with batching and helpful error messages.
"""

import logging
from functools import cached_property
from itertools import batched

import ollama

from docsgroundtruth.domain import DocsGroundTruthError

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Implements the `Embedder` interface (see domain.py) with a model served by Ollama."""

    def __init__(
        self,
        host: str,
        model: str,
        document_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 16,
    ) -> None:
        """
        Args:
            host: URL of the Ollama server, e.g. http://localhost:11434.
            model: Embedding model name, e.g. "nomic-embed-text".
            document_prefix: Text put in front of every document chunk (model-specific).
            query_prefix: Text put in front of every search query (model-specific).
            batch_size: How many texts to send to Ollama in one request.
        """
        self._host = host
        self._model = model
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._client = ollama.Client(host=host)

    @cached_property
    def dimension(self) -> int:
        """Vector length, measured once by embedding a throwaway text and then cached.

        Measuring it instead of hard-coding 768 means switching to a different
        embedding model needs no code change.
        """
        return len(self._embed(["dimension probe"])[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks in batches; returns vectors in the same order as `texts`."""
        prefixed = [self._document_prefix + text for text in texts]
        vectors: list[list[float]] = []
        # batched() yields groups of `batch_size` items: one HTTP request per group
        # instead of one per chunk, which is much faster for big documents.
        for batch in batched(prefixed, self._batch_size):
            vectors.extend(self._embed(list(batch)))
            logger.debug("Embedded %d/%d chunks", len(vectors), len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one search question."""
        return self._embed([self._query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """The only method that talks to Ollama, so all error handling lives in one place."""
        try:
            response = self._client.embed(model=self._model, input=texts)
        except ConnectionError as error:
            raise DocsGroundTruthError(
                f"Cannot reach Ollama at {self._host}. Is it running? "
                "(Check for the llama icon in the system tray, or run `ollama list`.)"
            ) from error
        except ollama.ResponseError as error:
            if error.status_code == 404:
                raise DocsGroundTruthError(
                    f"Embedding model '{self._model}' is not downloaded. "
                    f"Run: ollama pull {self._model}"
                ) from error
            raise
        return [list(vector) for vector in response.embeddings]
