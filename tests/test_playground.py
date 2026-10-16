"""Smoke test for the Streamlit playground: the page must load and render without errors.

Streamlit's AppTest runs the app script inside the test, with no browser and
no server. "Smoke test" means it checks the basics work, not every detail.
"""

from collections.abc import Iterator

import pytest
from streamlit.testing.v1 import AppTest

from docsgroundtruth.config import get_settings
from docsgroundtruth.ui import PLAYGROUND_SCRIPT


@pytest.fixture
def empty_index(tmp_path, monkeypatch) -> Iterator[None]:
    """Point the app at a fresh, empty index folder instead of the real data/index."""
    monkeypatch.setenv("DGT_QDRANT_PATH", str(tmp_path / "index"))
    get_settings.cache_clear()  # settings are cached, so rebuild them with the new variable
    yield
    get_settings.cache_clear()  # don't leak the test's settings into other tests


def test_playground_renders_and_explains_an_empty_index(empty_index):
    # The first run imports our package and Qdrant, which can take a few seconds.
    app = AppTest.from_file(str(PLAYGROUND_SCRIPT), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "DocsGroundTruth playground"
    assert "ingest" in app.sidebar.info[0].value  # "Nothing is indexed yet. Run ... ingest"
