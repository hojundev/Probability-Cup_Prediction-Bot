"""
View your SETTLED results for a match (or any question filter).

Lists your settled predictions whose match name (when the API provides it) or
question text contains the search term, with the recovered outcome and Brier
score. Settled markets aren't returned by /markets (open only), so this reads
the /results endpoint directly.

Usage:
    python3 results.py Germany
    python3 results.py Paraguay
    python3 results.py "Kai Havertz"

Caveat: /results doesn't always include the match name, so generic questions
that don't mention a team ("total cards", "both teams score", a hydration-break
goal) may not match a team-name search. Search by team name to catch the most.
"""

import sys
import logging

logging.basicConfig(level=logging.WARNING)

from bot.client import get_probability_cup_lobby_and_event, join_lobby, fetch_results


def _recover_outcome(prob_int, brier):
    """outcome from submitted prob (1-99) and Brier = (p-o)^2; None at p=50."""
    p = prob_int / 100.0
    d1 = abs(brier - (1 - p) ** 2)   # outcome == 1
    d0 = abs(brier - p ** 2)         # outcome == 0
    if abs(d1 - d0) < 1e-12:
        return None
    return 1 if d1 < d0 else 0


def _match_name(r):
    m = r.get("match")
    if isinstance(m, dict):
        return m.get("name", "") or ""
    return r.get("match_name", "") or ""


def main():
    term = " ".join(sys.argv[1:]).strip()
    if not term:
        print('Usage: python3 results.py Germany')
        return

    event, lobby = get_probability_cup_lobby_and_event()
    join_lobby(lobby["id"])
    results = fetch_results(lobby["id"])
    if not results:
        print("No settled predictions yet.")
        return

    t = term.lower()
    matched = [r for r in results
               if t in f"{_match_name(r)} {r.get('question', '')}".lower()]

    if not matched:
        print(f"No settled markets matching '{term}'.")
        print("(/results may omit the match name, so generic questions without a")
        print(" team name won't match — try a single team, e.g. 'Germany'.)")
        return

    print(f"\nSettled results matching '{term}'  ({len(matched)} markets)")
    print(f"{'Question':<66}{'Pred':>6}{'Out':>5}{'Brier':>8}")
    print("-" * 85)

    total_brier, n = 0.0, 0
    for r in sorted(matched, key=lambda x: x.get("question", "")):
        prob = r.get("probability_submitted", r.get("probability"))
        brier = r.get("brier_score")
        q = (r.get("question", "") or "")[:64]
        if prob is None or brier is None:
            print(f"{q:<66}{str(prob):>6}{'?':>5}{'n/a':>8}")
            continue
        o = _recover_outcome(prob, brier)
        out = "YES" if o == 1 else ("NO" if o == 0 else "?")
        print(f"{q:<66}{prob:>5}%{out:>5}{brier:>8.3f}")
        total_brier += brier
        n += 1

    if n:
        print("-" * 85)
        print(f"{'mean Brier':<66}{'':>6}{'':>5}{total_brier / n:>8.3f}")
        print("(lower Brier = better; YES/NO is the actual outcome.)")


if __name__ == "__main__":
    main()
