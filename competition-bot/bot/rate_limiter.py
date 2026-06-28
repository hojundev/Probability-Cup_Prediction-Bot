"""
Central rate limiter for the SportsPredict API.

SportsPredict enforces a hard limit of 60 requests per minute per IP. A classic
token bucket is the WRONG tool for a hard rolling-window limit: it allows a
burst of `capacity` plus a full window of refills (~2*capacity) inside a single
window, which would breach 60/min during a submission burst (200+ markets).

So this is a sliding-window log limiter: it records the completion time of every
acquire and guarantees that no more than `capacity` acquires complete in ANY
rolling window of `refill_seconds`. The default (55 per 60 s) leaves headroom
under the 60/min ceiling for clock skew and server-side counting differences.

The class keeps the name `TokenBucket` for backwards compatibility with existing
call sites. clock and sleep are injectable so tests run deterministically.
"""

import time as _time
import threading
from collections import deque


class TokenBucket:
    """
    Thread-safe sliding-window rate limiter.

    Invariant: at most `capacity` acquire() calls complete in any rolling window
    of `refill_seconds` seconds.

    capacity       – max completed acquires per window (also the burst ceiling)
    refill_seconds – length of the rolling window
    clock          – callable returning monotonic time in seconds
    sleep          – callable(seconds) used when blocking for a free slot
    """

    def __init__(
        self,
        capacity: float = 55,
        refill_seconds: float = 60,
        *,
        clock=_time.monotonic,
        sleep=_time.sleep,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}")
        if refill_seconds <= 0:
            raise ValueError(f"refill_seconds must be positive, got {refill_seconds!r}")
        self._capacity = capacity
        self._window = refill_seconds
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        # Completion timestamps still inside the current window, oldest first.
        self._events = deque()

    def _evict(self, now) -> None:
        """Drop timestamps that have aged out of the (now - window, now] window."""
        cutoff = now - self._window
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def acquire(self) -> None:
        """Block until a slot is free in the current window, then consume it."""
        while True:
            with self._lock:
                now = self._clock()
                self._evict(now)
                if len(self._events) < self._capacity:
                    self._events.append(now)
                    return
                # Full: wait until the oldest in-window event ages out, freeing
                # a slot. After it expires, (now' - window) >= oldest so _evict
                # will drop it on the next pass.
                wait = self._events[0] + self._window - now
            self._sleep(wait if wait > 0 else 0)
