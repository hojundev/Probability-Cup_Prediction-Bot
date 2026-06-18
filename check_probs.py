"""Quick probability lookup — reads only from local cache, no API calls."""
import sys
sys.path.insert(0, '/Users/jun/Desktop/Personal_Project/ProbabilityCup')

from data.fetch_odds import fetch_market_odds
from bot.match_data import build_odds_index, find_match_odds, estimate_team_xg
from model.poisson import (
    predict_btts, prob_at_least_one, prob_total_goals, halftime_outcome_probs
)
from model.ensemble import format_prediction_for_submission

odds_index = build_odds_index(fetch_market_odds())

def show(match_name, team_a, team_b):
    odds = find_match_odds(odds_index, match_name)
    if not odds:
        print(f"No odds found for {match_name}")
        return
    xg_h, xg_a = estimate_team_xg(
        odds['p_home'], odds['p_draw'], odds['p_away'], odds.get('total_goals')
    )
    ht_home, ht_tie, ht_away = halftime_outcome_probs(xg_h, xg_a)
    btts = predict_btts(xg_h, xg_a)

    p = format_prediction_for_submission
    rows = [
        (f"Will {team_a} win the match?",               p(odds['p_home'])),
        (f"Will {team_b} win the match?",               p(odds['p_away'])),
        ("Draw",                                         p(odds['p_draw'])),
        (f"Will {team_a} score at least 1 goal?",       p(prob_at_least_one(xg_h))),
        (f"Will {team_b} score at least 1 goal?",       p(prob_at_least_one(xg_a))),
        ("Will both teams score (BTTS)?",               p(btts)),
        ("Will the match have 2 or more total goals?",  p(prob_total_goals(xg_h, xg_a, 2, 'over'))),
        ("Will the match have 3 or more total goals?",  p(prob_total_goals(xg_h, xg_a, 3, 'over'))),
        ("Will the match have 2 or fewer total goals?", p(prob_total_goals(xg_h, xg_a, 2, 'under'))),
        ("At halftime, will the match be tied?",        p(ht_tie)),
        (f"At halftime, will {team_a} be winning?",     p(ht_home)),
        (f"At halftime, will {team_b} be winning?",     p(ht_away)),
        (f"Will {team_a} score in the second half?",    p(prob_at_least_one(xg_h * 0.58))),
        (f"Will {team_b} score in the second half?",    p(prob_at_least_one(xg_a * 0.58))),
        ("Will a penalty OR red card be shown?",        40),
    ]

    print(f"\n=== {match_name} ===")
    print(f"  Odds: {team_a} win={odds['p_home']*100:.1f}%  "
          f"Draw={odds['p_draw']*100:.1f}%  {team_b} win={odds['p_away']*100:.1f}%")
    print(f"  Goals line: {odds['total_goals']:.2f}  |  "
          f"xG: {team_a}={xg_h:.2f}  {team_b}={xg_a:.2f}\n")
    print(f"  {'Question':<55} {'Prob':>4}")
    print("  " + "-"*61)
    for q, prob in rows:
        print(f"  {q:<55} {prob:>4}")

show('CZE vs RSA', 'Czechia', 'South Africa')
show('SUI vs BIH', 'Switzerland', 'Bosnia-Herzegovina')
