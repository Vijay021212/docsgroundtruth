"""
Reranking: a careful second look at the best candidates, powered by fastembed.

Our embedding model reads the question and each chunk SEPARATELY and compares
the resulting numbers. A reranker (a "cross-encoder") reads the question and
ONE chunk TOGETHER, and judges whether that chunk actually answers the question.
That's much more accurate, but it must run once per chunk for every question,
so the Retriever only hands it the top candidates from hybrid search.

The model is an ONNX file, run by ONNX Runtime on the CPU (no GPU needed).
"""

import math
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder


class FastEmbedReranker:
    """Implements the `Reranker` interface (see domain.py) with a fastembed cross-encoder."""

    def __init__(self, model: str, cache_dir: Path) -> None:
        """
        Args:
            model: A reranker model name from fastembed's supported list.
            cache_dir: Folder where fastembed keeps downloaded models.
        """
        # The model file is downloaded here the first time (80 MB for the default).
        # lazy_load=True waits until the first question to load it into memory, so
        # creating a reranker stays quick for searches that never rerank.
        self._model = TextCrossEncoder(model_name=model, cache_dir=str(cache_dir), lazy_load=True)

    def score(self, question: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the question, from 0 (unrelated) to 1 (answers it).

        The model reads at most 512 tokens per question + text pair (for the default
        model). fastembed cuts longer texts automatically, keeping their beginning,
        which is where our chunks carry their section headings.
        """
        # The model returns raw scores ("logits") such as -10.6 or 4.2, which are hard
        # to read. sigmoid() squeezes any number into 0..1: large positive -> near 1.
        return [sigmoid(raw) for raw in self._model.rerank(question, texts)]


def sigmoid(x: float) -> float:
    """Map any number into the range 0..1 (0 maps to 0.5)."""
    # Two forms of the same formula, chosen so math.exp never gets a large positive
    # number. That would overflow for very confident scores like -1000.
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)
