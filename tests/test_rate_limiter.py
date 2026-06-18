"""
tests/test_rate_limiter.py
--------------------------
Unit and property-based tests for TokenBucket (bot/rate_limiter.py).

All tests use injected clock/sleep so no real time is consumed.

**Validates: Requirements 6.2**
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hypothesis import given, settings, assume
from hypothesis import strategies as st
import pytest

from bot.rate_limiter import TokenBucket


# ---------------------------------------------------------------------------
# Helpers: deterministic simulation infrastructure
# ---------------------------------------------------------------------------

class FakeClock:
    """A fake monotonic clock whose value is advanced manually."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingSleep:
    """A fake sleep that simply advances the clock it wraps."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.total_slept = 0.0

    def __call__(self, seconds: float) -> None:
        assert seconds >= 0, f"sleep called with negative duration {seconds}"
        self._clock.advance(seconds)
        self.total_slept += seconds


def make_bucket(capacity: float = 10, refill_seconds: float = 10,
                start: float = 0.0) -> tuple["TokenBucket", FakeClock]:
    """Convenience factory returning a bucket and its fake clock."""
    clock = FakeClock(start)
    sleep = RecordingSleep(clock)
    bucket = TokenBucket(capacity=capacity, refill_seconds=refill_seconds,
                         clock=clock, sleep=sleep)
    return bucket, clock


# ---------------------------------------------------------------------------
# Unit tests: construction guards
# ---------------------------------------------------------------------------

class TestTokenBucketConstruction:
    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(capacity=0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(capacity=-1)

    def test_zero_refill_seconds_raises(self):
        with pytest.raises(ValueError, match="refill_seconds"):
            TokenBucket(capacity=10, refill_seconds=0)

    def test_negative_refill_seconds_raises(self):
        with pytest.raises(ValueError, match="refill_seconds"):
            TokenBucket(capacity=10, refill_seconds=-5)


# ---------------------------------------------------------------------------
# Unit tests: immediate pass-through when tokens are available
# ---------------------------------------------------------------------------

class TestImmediatePassThrough:
    def test_first_acquire_does_not_sleep(self):
        clock = FakeClock()
        sleep = RecordingSleep(clock)
        bucket = TokenBucket(capacity=5, refill_seconds=5, clock=clock, sleep=sleep)

        bucket.acquire()

        assert sleep.total_slept == 0.0, "Should not sleep when bucket starts full"

    def test_capacity_acquires_complete_without_sleeping(self):
        """All capacity tokens can be consumed at t=0 without any sleep."""
        cap = 8
        clock = FakeClock()
        sleep = RecordingSleep(clock)
        bucket = TokenBucket(capacity=cap, refill_seconds=10, clock=clock, sleep=sleep)

        for _ in range(cap):
            bucket.acquire()

        assert sleep.total_slept == 0.0

    def test_time_does_not_advance_for_available_token(self):
        """Acquiring a token when one is available must not advance the fake clock."""
        bucket, clock = make_bucket(capacity=3, refill_seconds=3)
        t_before = clock.now
        bucket.acquire()
        assert clock.now == t_before


# ---------------------------------------------------------------------------
# Unit tests: blocking when empty
# ---------------------------------------------------------------------------

class TestBlockingWhenEmpty:
    def test_acquire_blocks_when_empty(self):
        """After draining the bucket, the next acquire must sleep."""
        cap = 3
        bucket, clock = make_bucket(capacity=cap, refill_seconds=3)
        sleep = RecordingSleep(clock)
        bucket._sleep = sleep  # wire in the recording sleep

        # Drain all tokens at t=0 with no clock advance.
        for _ in range(cap):
            bucket.acquire()

        # The next acquire must block (sleep > 0).
        t_before = clock.now
        bucket.acquire()
        assert clock.now > t_before, "acquire should have advanced time when bucket was empty"

    def test_acquire_waits_at_most_one_token_interval(self):
        """
        After a full drain, the wait for one token is at most 1 / rate seconds
        (one full token-refill period).
        """
        cap = 5
        refill = 10.0
        bucket, clock = make_bucket(capacity=cap, refill_seconds=refill)
        rate = cap / refill  # tokens per second

        # Drain.
        for _ in range(cap):
            bucket.acquire()

        t_before = clock.now
        bucket.acquire()
        waited = clock.now - t_before

        # One token takes 1/rate seconds to arrive; allow tiny float tolerance.
        assert waited <= (1.0 / rate) + 1e-9


# ---------------------------------------------------------------------------
# Unit tests: refill timing
# ---------------------------------------------------------------------------

class TestRefillTiming:
    def test_tokens_accumulate_over_time(self):
        """Tokens refill at the expected rate."""
        cap = 10
        refill = 10.0  # rate = 1 token/sec
        bucket, clock = make_bucket(capacity=cap, refill_seconds=refill)

        # Drain all tokens.
        for _ in range(cap):
            bucket.acquire()

        # Advance exactly 5 seconds — should have 5 tokens.
        clock.advance(5.0)
        bucket._refill()
        assert abs(bucket._tokens - 5.0) < 1e-9

    def test_tokens_do_not_exceed_capacity(self):
        """Refill must not overfill beyond capacity."""
        cap = 10
        bucket, clock = make_bucket(capacity=cap, refill_seconds=10)

        # Start full; advance a huge amount.
        clock.advance(1_000_000)
        bucket._refill()
        assert bucket._tokens == cap

    def test_partial_refill(self):
        """Half the refill period restores half the capacity from empty."""
        cap = 20
        refill = 20.0  # rate = 1 token/sec
        bucket, clock = make_bucket(capacity=cap, refill_seconds=refill)

        # Drain.
        for _ in range(cap):
            bucket.acquire()

        clock.advance(10.0)  # 10 seconds → 10 tokens
        bucket._refill()
        assert abs(bucket._tokens - 10.0) < 1e-9

    def test_two_acquires_separated_by_one_interval(self):
        """Acquiring, waiting one token's worth of time, then acquiring again."""
        cap = 5
        refill = 5.0  # rate = 1 token/sec; 1 token per second
        bucket, clock = make_bucket(capacity=cap, refill_seconds=refill)

        # Drain completely.
        for _ in range(cap):
            bucket.acquire()

        # Advance exactly 1 second — exactly 1 token should have arrived.
        clock.advance(1.0)

        # The next acquire should succeed without extra sleep.
        sleep = RecordingSleep(clock)
        bucket._sleep = sleep
        bucket.acquire()
        assert sleep.total_slept == 0.0, "One token should have been ready after 1 s"


# ---------------------------------------------------------------------------
# Property-based tests: the core invariant
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=20),
    refill_seconds=st.floats(min_value=1.0, max_value=60.0, allow_nan=False,
                             allow_infinity=False),
    n_acquires=st.integers(min_value=1, max_value=30),
    time_between=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=0, max_size=30,
    ),
)
def test_no_more_than_capacity_in_any_window(
    capacity, refill_seconds, n_acquires, time_between
):
    """
    **Validates: Requirements 6.2**

    Invariant: in any rolling window of length `refill_seconds` the total
    number of completed acquire() calls must not exceed `capacity`.

    The test drives the bucket through `n_acquires` calls with optional
    idle gaps between them (drawn from `time_between`).  After all calls
    complete it scans every consecutive sub-window of width `refill_seconds`
    in the recorded completion timestamps and asserts the count ≤ capacity.
    """
    assume(refill_seconds > 0)

    clock = FakeClock(start=0.0)
    sleep = RecordingSleep(clock)
    bucket = TokenBucket(
        capacity=capacity,
        refill_seconds=refill_seconds,
        clock=clock,
        sleep=sleep,
    )

    # Interleave manual clock advances with acquire() calls to simulate
    # varied arrival patterns (bursty, steady, staggered).
    timestamps: list[float] = []
    for i in range(n_acquires):
        # Optionally advance the clock before this acquire (simulate idle time).
        if i < len(time_between):
            clock.advance(time_between[i])
        bucket.acquire()
        timestamps.append(clock.now)

    # Verify the invariant: no window of `refill_seconds` contains > capacity completions.
    for i, t_start in enumerate(timestamps):
        window_end = t_start + refill_seconds
        count = sum(1 for t in timestamps if t_start <= t < window_end)
        assert count <= capacity, (
            f"Window [{t_start:.4f}, {window_end:.4f}) contains {count} completions "
            f"but capacity={capacity}.  timestamps={timestamps}"
        )


@settings(max_examples=100, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=20),
    refill_seconds=st.floats(min_value=0.5, max_value=60.0, allow_nan=False,
                             allow_infinity=False),
    burst_size=st.integers(min_value=1, max_value=50),
)
def test_burst_then_drain_respects_capacity(capacity, refill_seconds, burst_size):
    """
    **Validates: Requirements 6.2**

    A burst of acquire() calls (no idle time between them) must never allow
    more than `capacity` completions in any `refill_seconds` window, even
    when the burst size exceeds capacity.
    """
    assume(refill_seconds > 0)

    clock = FakeClock(start=0.0)
    sleep = RecordingSleep(clock)
    bucket = TokenBucket(
        capacity=capacity,
        refill_seconds=refill_seconds,
        clock=clock,
        sleep=sleep,
    )

    timestamps: list[float] = []
    for _ in range(burst_size):
        bucket.acquire()
        timestamps.append(clock.now)

    for i, t_start in enumerate(timestamps):
        window_end = t_start + refill_seconds
        count = sum(1 for t in timestamps if t_start <= t < window_end)
        assert count <= capacity, (
            f"Burst window [{t_start:.4f}, {window_end:.4f}) contains {count} "
            f"completions but capacity={capacity}. timestamps={timestamps}"
        )


@settings(max_examples=100, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=20),
    refill_seconds=st.floats(min_value=0.5, max_value=60.0, allow_nan=False,
                             allow_infinity=False),
    idle_gaps=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=40,
    ),
)
def test_staggered_arrivals_respect_capacity(capacity, refill_seconds, idle_gaps):
    """
    **Validates: Requirements 6.2**

    Acquisitions separated by irregular idle times must still obey the
    capacity invariant.
    """
    assume(refill_seconds > 0)

    clock = FakeClock(start=0.0)
    sleep = RecordingSleep(clock)
    bucket = TokenBucket(
        capacity=capacity,
        refill_seconds=refill_seconds,
        clock=clock,
        sleep=sleep,
    )

    timestamps: list[float] = []
    for gap in idle_gaps:
        clock.advance(gap)
        bucket.acquire()
        timestamps.append(clock.now)

    for t_start in timestamps:
        window_end = t_start + refill_seconds
        count = sum(1 for t in timestamps if t_start <= t < window_end)
        assert count <= capacity, (
            f"Staggered window [{t_start:.4f}, {window_end:.4f}) contains {count} "
            f"completions but capacity={capacity}. timestamps={timestamps}"
        )


# ---------------------------------------------------------------------------
# Standalone runner (optional convenience)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], check=True)
