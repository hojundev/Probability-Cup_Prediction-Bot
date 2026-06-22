"""
Central token-bucket rate limiter for the SportsPredict API.

A continuous-refill bucket is used instead of fixed sleeps so every call site
shares one budget and pacing self-corrects.  The default capacity is
intentionally conservative (55 tokens per 60 s) to leave headroom for
clock-skew and server-side counting differences.

clock and sleep are injectable so tests can run deterministically.
"""

import time as _time
import threading


class TokenBucket:
    """
    Thread-safe continuous-refill token bucket.

    capacity      – maximum tokens (also the burst ceiling)
    refill_seconds – window over which `capacity` tokens fully refill
    clock         – callable returning monotonic time in seconds
    sleep         – callable(seconds) used when blocking for a token
    """

    def __init__(
        self,
        capacity: float = 55,
        refill_seconds: float = 60,
        *,
        clock=_time.monotonic,
        sleep=_time.sleep,
    ):
        self._capacity = capacity
        self._rate = capacity / refill_seconds      # tokens per second
        self._tokens = capacity                     # start full
        self._last = clock()
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Add tokens earned since the last call (called under lock)."""
        now = self._clock()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate exact wait for the next token
                wait = (1.0 - self._tokens) / self._rate
            self._sleep(wait)
