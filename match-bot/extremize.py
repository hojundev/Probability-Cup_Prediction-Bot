"""
Selective extremizing for the match-bot.

The match-bot's objective is to win per-match leaderboard prizes (top 1 of a
single match's markets), not to minimize overall Brier across the competition.
Winning a single match doesn't require calibration — it requires being
confident and right. So we push the competition-bot's probabilities further
from 0.50 via a logit stretch on the markets where the model has genuine signal.
"""

import math


def extremize(p, k=1.4):
    """
    Push probability ``p`` further from 0.5 using a logit stretch.

    ``k`` controls aggression:
      - k = 1.0 -> identical to the input (no change)
      - k = 1.3 -> mild extremizing
      - k = 1.4-1.5 -> recommended starting range
      - k = 2.0+ -> very aggressive, high variance

    Examples with k=1.4:
      p=0.60 -> ~0.67
      p=0.70 -> ~0.79
      p=0.80 -> ~0.88
      p=0.40 -> ~0.33
      p=0.30 -> ~0.21

    The transform is symmetric about 0.5, monotonic, and fixes the endpoints
    0, 0.5, and 1. Values at or beyond the [0, 1] boundary are returned
    unchanged (their logit is undefined).
    """
    if p <= 0 or p >= 1:
        return p
    logit = math.log(p / (1 - p))
    stretched = k * logit
    return 1 / (1 + math.exp(-stretched))
