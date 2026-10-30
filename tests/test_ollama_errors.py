"""Tests for the clear error messages about Ollama problems."""

import httpx
import ollama
import pytest

from docsgroundtruth.domain import DocsGroundTruthError
from docsgroundtruth.ollama_errors import explain_ollama_errors


def _raise_inside(error: Exception) -> None:
    """Raise `error` inside the block whose errors get explained."""
    with explain_ollama_errors("http://localhost:11434", "some-model"):
        raise error


# Normal requests raise ConnectionError; streaming requests raise httpx.ConnectError.
@pytest.mark.parametrize("error", [ConnectionError("down"), httpx.ConnectError("down")])
def test_a_failed_connection_is_explained(error):
    with pytest.raises(DocsGroundTruthError, match="Cannot reach Ollama"):
        _raise_inside(error)


def test_a_missing_model_tells_you_how_to_download_it():
    with pytest.raises(DocsGroundTruthError, match="ollama pull some-model"):
        _raise_inside(ollama.ResponseError("not found", 404))


def test_other_ollama_errors_are_not_hidden():
    with pytest.raises(ollama.ResponseError):
        _raise_inside(ollama.ResponseError("server error", 500))
