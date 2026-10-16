"""Tests for the Stopwatch timing helper."""

import time

import pytest

from docsgroundtruth.timing import Stopwatch


def test_stopwatch_measures_how_long_the_block_took():
    with Stopwatch() as watch:
        time.sleep(0.05)

    # A slightly lower bound than 0.05 keeps the test reliable: on Windows the
    # sleep timer and the stopwatch clock can disagree by a fraction of a millisecond.
    assert watch.seconds >= 0.04


def test_stopwatch_still_records_the_time_when_the_block_fails():
    watch = Stopwatch()

    with pytest.raises(RuntimeError), watch:
        time.sleep(0.01)
        raise RuntimeError("boom")

    assert watch.seconds > 0
