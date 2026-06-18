"""
Offline probability report for Uzbekistan vs Colombia.
Uses cached odds (if available) or reasonable fallback xG values.
Run: python3 -m tests.test_uzb_col
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.fetch_odds import fetch_market_odds
from bot.match_data import build_odds_index, find_match_odds, estimate_team_xg
from model.poisson import (
    predict_match_outcome, predict_btts, prob_at_least_one,
    prob_over_under, prob_total_goals, halftime_outcome_probs,
)
from model.ensemble import format_prediction_for_submission

MATCH = "UZB vs COL"  # SportsPredict format

def main():
    # Try live/cached odds first
    odds_list = fetch_market_odds()
    idx = build_odds_index(odds_list)
    odds = find_match_odds(idx, MATCH)

    if odds:
        print(f"Odds source: cached/live")
        print(f"  p_home(UZB)={odds['p_home']:.3f}  p_draw={odds['p_draw']:.3f}  p_away(COL)={odds['p_away']:.3f}")
        xg_uzb, xg_col = estimate_team_xg(
            odds["p_home"], odds["p_draw"], odds["p_away"], odds.get("total_goals")
        )
    else:
        print("No odds found for UZB vs COL — using fallback xG (Colombia strong favourite)")
        # Colombia is a strong favourite; Uzbekistan is a heavy underdog at the WC
        xg_uzb, xg_col = 0.55, 1.85
        odds = {"p_home": 0.14, "p_draw": 0.22, "p_away": 0.64}

    print(f"  xG  UZB={xg_uzb:.3f}  COL={xg_col:.3f}\n")

    outcome = predict_match_outcome(xg_uzb, xg_col)
    ht_home, ht_tie, ht_away = halftime_outcome_probs(xg_uzb, xg_col)

    rows = [
        ("UZB win",              outcome["home_win"]),
        ("Draw",                 outcome["draw"]),
        ("COL win",              outcome["away_win"]),
        ("UZB scores",           prob_at_least_one(xg_uzb)),
        ("COL scores",           prob_at_least_one(xg_col)),
        ("BTTS",                 predict_btts(xg_uzb, xg_col)),
        ("Over 2.5 goals",       prob_total_goals(xg_uzb, xg_col, 2.5, "over")),
        ("Under 2.5 goals",      prob_total_goals(xg_uzb, xg_col, 2.5, "under")),
        ("Over 1.5 goals",       prob_total_goals(xg_uzb, xg_col, 1.5, "over")),
        ("HT: UZB leading",      ht_home),
        ("HT: draw",             ht_tie),
        ("HT: COL leading",      ht_away),
    ]

    print(f"{'Market':<30} {'P(yes)':>8}   {'1-99':>5}")
    print("-" * 50)
    for label, p in rows:
        print(f"{label:<30} {p:>8.3f}   {format_prediction_for_submission(p):>5}")

if __name__ == "__main__":
    main()
