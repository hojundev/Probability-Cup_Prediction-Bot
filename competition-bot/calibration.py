"""
Calibration / backtest report from settled predictions.

Pulls your settled predictions (GET /results), recovers each binary outcome from
the Brier score, and reports how well-calibrated the bot is — overall, per
question type, and per prediction band. Use it to decide WHERE the model is
over/under-predicting and WHETHER you have enough resolved markets to act on it.

How outcomes are recovered: the API gives `probability_submitted` (1-99) and
`brier_score = (p - outcome)^2` per settled market. Since outcome is 0 or 1:
    outcome = 1  if brier ≈ (1 - p)^2
    outcome = 0  if brier ≈ p^2
(p = probability_submitted / 100). A prediction of exactly 50 is indistinguishable
and is reported as "undetermined".

Note: the API exposes YOUR Brier but not the crowd's, so this is calibration
(prediction vs actual), not RBP vs crowd. Calibration is the right signal for
"is the model biased high or low" regardless.

Usage:
    python3 calibration.py            # all settled markets
    python3 calibration.py player     # only types whose name contains "player"
"""

import sys
import logging
from collections import defaultdict

logging.basicConfig(level=logging.WARNING)

from bot.client import get_probability_cup_lobby_and_event, join_lobby, fetch_results
from bot.question_parser import parse_question

MIN_SAMPLE = 30   # below this, a type is too thin to tune on


def _recover_outcome(prob_int, brier):
    """Recover the 0/1 outcome from submitted prob (1-99) and Brier = (p-o)^2."""
    p = prob_int / 100.0
    d1 = abs(brier - (1 - p) ** 2)   # outcome == 1
    d0 = abs(brier - p ** 2)         # outcome == 0
    if abs(d1 - d0) < 1e-12:         # p == 0.50 -> indistinguishable
        return None
    return 1 if d1 < d0 else 0


def _pct(x):
    return f"{100 * x:5.1f}%" if x is not None else "  n/a"


def main():
    type_filter = " ".join(sys.argv[1:]).strip().lower()

    event, lobby = get_probability_cup_lobby_and_event()
    join_lobby(lobby["id"])
    results = fetch_results(lobby["id"])

    if not results:
        print("No settled predictions yet — nothing to calibrate.")
        return

    rows = []          # (qtype, prob_int, outcome, brier)
    undetermined = 0
    for r in results:
        prob = r.get("probability_submitted", r.get("probability"))
        brier = r.get("brier_score")
        if prob is None or brier is None:
            continue
        qtype = parse_question(r.get("question", "")).get("type", "unknown")
        if type_filter and type_filter not in qtype:
            continue
        outcome = _recover_outcome(prob, brier)
        if outcome is None:
            undetermined += 1
            continue
        rows.append((qtype, prob, outcome, brier))

    if not rows:
        print("No determinable settled outcomes.")
        return

    n = len(rows)
    mean_pred = sum(r[1] for r in rows) / n / 100.0
    hit_rate = sum(r[2] for r in rows) / n
    mean_brier = sum(r[3] for r in rows) / n

    print("=" * 72)
    print(f"OVERALL   ({n} settled markets, {undetermined} undetermined @ p=50)")
    print("=" * 72)
    print(f"  mean prediction   : {_pct(mean_pred)}")
    print(f"  actual hit rate   : {_pct(hit_rate)}")
    bias = mean_pred - hit_rate
    print(f"  bias (pred-actual): {100 * bias:+.1f} pts  "
          f"({'OVER-predicting' if bias > 0 else 'under-predicting'})")
    print(f"  mean Brier        : {mean_brier:.4f}")

    # Per question type
    by_type = defaultdict(list)
    for qtype, prob, outcome, brier in rows:
        by_type[qtype].append((prob, outcome, brier))

    print("\n" + "=" * 72)
    print("BY QUESTION TYPE   (bias = mean prediction - actual hit rate)")
    print("=" * 72)
    print(f"  {'type':<26}{'n':>4}{'pred':>8}{'actual':>9}{'bias':>8}{'Brier':>9}")
    print("  " + "-" * 64)
    for qtype in sorted(by_type, key=lambda t: -len(by_type[t])):
        items = by_type[qtype]
        m = len(items)
        pred = sum(i[0] for i in items) / m / 100.0
        act = sum(i[1] for i in items) / m
        br = sum(i[2] for i in items) / m
        flag = "" if m >= MIN_SAMPLE else "  (thin)"
        print(f"  {qtype:<26}{m:>4}{_pct(pred):>8}{_pct(act):>9}"
              f"{100 * (pred - act):>+7.1f}{br:>9.3f}{flag}")

    # Calibration by prediction band
    bands = [(1, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 100)]
    print("\n" + "=" * 72)
    print("CALIBRATION BY PREDICTION BAND   (well-calibrated: pred ≈ actual)")
    print("=" * 72)
    print(f"  {'band':<10}{'n':>5}{'mean pred':>12}{'actual':>10}")
    print("  " + "-" * 37)
    for lo, hi in bands:
        items = [r for r in rows if lo <= r[1] < hi]
        if not items:
            continue
        m = len(items)
        pred = sum(i[1] for i in items) / m / 100.0
        act = sum(i[2] for i in items) / m
        print(f"  {f'{lo}-{hi - 1}':<10}{m:>5}{_pct(pred):>12}{_pct(act):>10}")

    print("\nNotes:")
    print("  - bias > 0  => predicted higher than reality (over-predicting); lower the prior.")
    print(f"  - types with n < {MIN_SAMPLE} are flagged '(thin)' — too few to tune on yet.")
    print("  - this is calibration vs actual outcomes, not RBP vs the crowd (crowd Brier")
    print("    isn't exposed by the API).")


if __name__ == "__main__":
    main()
