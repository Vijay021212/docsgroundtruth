"""Tests for the reranker's score conversion. The model itself is too heavy for unit tests."""

import pytest

from docsgroundtruth.reranking import sigmoid


def test_sigmoid_maps_any_score_into_zero_to_one():
    assert sigmoid(0) == 0.5
    assert sigmoid(10) == pytest.approx(1.0, abs=1e-4)
    assert sigmoid(-10) == pytest.approx(0.0, abs=1e-4)


def test_sigmoid_handles_extreme_scores_without_overflow():
    # math.exp(1000) would overflow; the two-form formula never computes it.
    assert sigmoid(-1000) == 0.0
    assert sigmoid(1000) == 1.0
