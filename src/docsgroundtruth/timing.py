"""
A tiny stopwatch for measuring how long each step of the pipeline takes.

Written once here so the "start the clock, stop the clock" code isn't repeated
in every module that reports timings.
"""

from __future__ import annotations

import time


class Stopwatch:
    """Measures how long the code inside a `with` block takes.

        with Stopwatch() as watch:
            do_something_slow()
        print(f"took {watch.seconds:.1f}s")
    """

    def __init__(self) -> None:
        self.seconds = 0.0  # filled in when the `with` block ends
        self._started = 0.0

    def __enter__(self) -> Stopwatch:
        # perf_counter() is Python's most precise clock for measuring durations.
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        # Runs when the block ends, even if it ended with an error.
        self.seconds = time.perf_counter() - self._started
