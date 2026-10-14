"""Clocks for the engine: a fake clock for tests and simulation, a wall clock for GPU runs.

Plumbing, not a numbered problem.

Operations
    Clock (protocol)
        now() -> float            current time in seconds
        advance_to(t) -> None     wait, or jump, until time t; a time already past is a no-op
        charge(seconds) -> None   account for the work of one step

    FakeClock(start=0.0)
        charge() moves time forward by exactly the given amount. Deterministic.
    WallClock()
        now() reads perf_counter from creation; advance_to sleeps; charge does nothing,
        because the real time already passed while the step ran.

Errors
    charge raises ValueError for a negative amount.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def advance_to(self, t: float) -> None: ...

    def charge(self, seconds: float) -> None: ...


class FakeClock:
    """Simulated time. Only advance_to and charge move it."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance_to(self, t: float) -> None:
        if t > self._now:
            self._now = t

    def charge(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"cannot charge negative time, got {seconds}")
        self._now += seconds


class WallClock:
    """Real time, measured from the moment of construction."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._start

    def advance_to(self, t: float) -> None:
        remaining = t - self.now()
        if remaining > 0:
            time.sleep(remaining)

    def charge(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"cannot charge negative time, got {seconds}")
