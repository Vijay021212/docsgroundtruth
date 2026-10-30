"""Tests for the Stopwatch timing helper."""

import time

import pytest

from docsgroundtruth.timing import StepTimer, Stopwatch


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


def test_step_timer_records_each_step_in_order():
    timer = StepTimer()

    with timer.step("first"):
        pass
    with timer.step("second"):
        pass

    assert list(timer.timings) == ["first", "second"]


def test_step_timer_does_not_record_a_failed_step():
    timer = StepTimer()

    with pytest.raises(RuntimeError), timer.step("broken"):
        raise RuntimeError("boom")

    assert timer.timings == {}
