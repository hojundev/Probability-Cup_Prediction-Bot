import os
import sys
import math

# Make the match-bot package importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from hypothesis import given, strategies as st

from extremize import extremize


# --- Formula outputs (k=1.4) ------------------------------------------------
# These are the exact values of the documented logit-stretch formula at k=1.4.
# (The strategy doc's illustrative table quotes rounder numbers that actually
# correspond to a steeper k ~= 1.75; the formula itself is the source of truth.)
@pytest.mark.parametrize("p, expected", [
    (0.60, 0.638),
    (0.70, 0.766),
    (0.80, 0.877),
    (0.40, 0.362),
    (0.30, 0.234),
])
def test_formula_values(p, expected):
    assert extremize(p, 1.4) == pytest.approx(expected, abs=0.005)


def test_cross_check_against_logit_formula():
    # Independently recompute the transform and confirm the implementation.
    for p in (0.55, 0.62, 0.73, 0.41, 0.28):
        logit = math.log(p / (1 - p))
        expected = 1 / (1 + math.exp(-1.4 * logit))
        assert extremize(p, 1.4) == pytest.approx(expected, abs=1e-12)


# --- Fixed points -----------------------------------------------------------
def test_fixed_points():
    assert extremize(0.5, 1.4) == pytest.approx(0.5)
    # k=1 is the identity transform.
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert extremize(p, 1.0) == pytest.approx(p)


def test_boundaries_returned_unchanged():
    assert extremize(0.0, 1.4) == 0.0
    assert extremize(1.0, 1.4) == 1.0
    assert extremize(-0.5, 1.4) == -0.5
    assert extremize(1.5, 1.4) == 1.5


# --- Property-based tests ---------------------------------------------------
PROBS = st.floats(min_value=1e-6, max_value=1 - 1e-6,
                  allow_nan=False, allow_infinity=False)
KS = st.floats(min_value=1.0, max_value=3.0,
               allow_nan=False, allow_infinity=False)


@given(p=PROBS, k=KS)
def test_output_is_valid_probability(p, k):
    out = extremize(p, k)
    assert 0.0 <= out <= 1.0


@given(p=PROBS, k=st.floats(min_value=1.0, max_value=3.0))
def test_pushes_away_from_half(p, k):
    # With k >= 1, the result is at least as far from 0.5 as the input, and on
    # the same side of 0.5.
    out = extremize(p, k)
    assert abs(out - 0.5) >= abs(p - 0.5) - 1e-9
    if p > 0.5:
        assert out >= p - 1e-9
    elif p < 0.5:
        assert out <= p + 1e-9


@given(p=PROBS, k=st.floats(min_value=1.01, max_value=3.0))
def test_symmetry_about_half(p, k):
    # extremize is symmetric: extremize(1-p) == 1 - extremize(p).
    assert extremize(1 - p, k) == pytest.approx(1 - extremize(p, k), abs=1e-9)


@given(
    p=st.floats(min_value=0.5 + 1e-3, max_value=1 - 1e-6),
    k=KS,
)
def test_monotonic_in_k(p, k):
    # For a fixed p > 0.5, a larger k yields a larger (more extreme) output.
    assert extremize(p, k + 0.5) >= extremize(p, k) - 1e-9


@given(
    k=KS,
    a=st.floats(min_value=1e-6, max_value=1 - 1e-6),
    b=st.floats(min_value=1e-6, max_value=1 - 1e-6),
)
def test_monotonic_in_p(k, a, b):
    # extremize is order-preserving in p (strictly increasing transform).
    lo, hi = sorted((a, b))
    assert extremize(lo, k) <= extremize(hi, k) + 1e-9


@given(p=PROBS)
def test_identity_at_k_one(p):
    assert extremize(p, 1.0) == pytest.approx(p, abs=1e-9)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
