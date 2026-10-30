"""
Clear error messages for problems talking to Ollama.

Both the embedder and the chat model use this, so each message is written once.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import ollama

from docsgroundtruth.domain import DocsGroundTruthError


@contextmanager
def explain_ollama_errors(host: str, model: str) -> Iterator[None]:
    """Turn Ollama's technical errors inside the `with` block into clear messages."""
    try:
        yield
    # Normal requests report a failed connection as ConnectionError, but streaming
    # requests let httpx's own ConnectError through, so we need to catch both.
    except (ConnectionError, httpx.ConnectError) as error:
        raise DocsGroundTruthError(
            f"Cannot reach Ollama at {host}. Is it running? "
            "(Check for the llama icon in the system tray, or run `ollama list`.)"
        ) from error
    except ollama.ResponseError as error:
        if error.status_code == 404:
            raise DocsGroundTruthError(
                f"Model '{model}' is not downloaded. Run: ollama pull {model}"
            ) from error
        raise
