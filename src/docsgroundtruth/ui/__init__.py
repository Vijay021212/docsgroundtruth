"""
Web interfaces for DocsGroundTruth.

`playground.py` is a Streamlit app for testing retrieval by hand. Streamlit
runs apps from a file path, so the path is defined here once and reused by the
`docsgroundtruth ui` command and by the tests.
"""

from pathlib import Path

PLAYGROUND_SCRIPT = Path(__file__).parent / "playground.py"
