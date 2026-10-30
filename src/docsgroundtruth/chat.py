"""
The chat model, which writes answers, run by the local Ollama server.

This is the "G" (generation) in RAG. It's the same kind of call as
`ollama run` in Phase 0, but with fixed settings and word-by-word streaming.
"""

import logging
from collections.abc import Iterator

import ollama

from docsgroundtruth.ollama_errors import explain_ollama_errors

logger = logging.getLogger(__name__)


class OllamaChatModel:
    """Implements the `ChatModel` interface (see domain.py) with a model served by Ollama."""

    def __init__(self, host: str, model: str, context_tokens: int, max_answer_tokens: int) -> None:
        """
        Args:
            host: URL of the Ollama server, e.g. http://localhost:11434.
            model: Chat model name, e.g. "qwen3:4b-instruct-2507-q4_K_M".
            context_tokens: How many tokens the model can see at once: instructions,
                sources and answer together. Ollama silently cuts off anything beyond
                this, so it must be set high enough for our sources to fit.
            max_answer_tokens: Upper limit on the length of an answer.
        """
        self._host = host
        self._model = model
        self._client = ollama.Client(host=host)
        self._options = {
            "num_ctx": context_tokens,
            "num_predict": max_answer_tokens,
            # 0 = always pick the most likely next word. We want factual, repeatable
            # answers, not creative ones, and repeatable answers are easier to evaluate.
            "temperature": 0.0,
        }

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the model's reply piece by piece, as it's being written."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        with explain_ollama_errors(self._host, self._model):
            for part in self._client.chat(
                model=self._model, messages=messages, stream=True, options=self._options
            ):
                if part.message.content:
                    yield part.message.content
                if part.done_reason == "length":
                    logger.warning("The answer reached the length limit and was cut off.")
