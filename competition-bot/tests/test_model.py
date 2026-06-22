import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.ensemble import format_prediction_for_submission, blend_probabilities
from model.poisson import predict_btts
from bot.question_parser import parse_question
from bot.match_data import (
    normalize_team_name,
    split_match_name,
    build_odds_index,
    find_match_odds,
    estimate_team_xg,
    team_is_home,
)
from bot.submit import run_model_on_market

def test_format_prediction():
    assert format_prediction_for_submission(0.5) == 50
    assert format_prediction_for_submission(0.0) == 1
    assert format_prediction_for_submission(1.0) == 99
    assert format_prediction_for_submission(0.999) == 99

def test_blend_probabilities():
    assert blend_probabilities(0.4, 0.6, alpha=0.5) == 0.5
    assert blend_probabilities(None, 0.6, alpha=0.5) == 0.6

def test_question_parser():
    assert parse_question("Will Ghana win the match?")["type"] == "match_winner"

    p = parse_question("Will Mexico score in the second half?")
    assert p["type"] == "team_score_half"
    assert p["team"] == "Mexico"
    assert p["half"] == "second"

    p = parse_question("Will the match have 3 or more total goals?")
    assert p["type"] == "total_goals"
    assert p["threshold"] == 3
    assert p["direction"] == "over"

    p = parse_question("Will the match have 2 or fewer total goals?")
    assert p["type"] == "total_goals"
    assert p["direction"] == "under"

    p = parse_question("Will Ghana have 3 or more shots on target?")
    assert p["type"] == "team_total_sot"
    assert p["threshold"] == 3

    p = parse_question("Will Antoine Semenyo have at least 1 shot on target?")
    assert p["type"] == "player_shot_on_target"
    assert p["player"] == "Antoine Semenyo"

    p = parse_question("Will Son score or assist a goal (excluding own goals)?")
    assert p["type"] == "player_goal_involvement"

    p = parse_question("Will Colombia have more shots on target than Uzbekistan in the second half?")
    assert p["type"] == "team_more_than_opponent"
    assert p["metric"] == "shots on target"
    assert p["half"] == "second"

    assert parse_question("At halftime, will the match be tied?")["type"] == "halftime_tied"
    assert parse_question("At halftime, will Canada be winning?")["type"] == "halftime_winning"
    assert parse_question("Will Uzbekistan have 5 or more corner kicks?")["type"] == "team_corners"
    assert parse_question("Will Panama be caught offside 2 or more times?")["type"] == "team_offsides"
    assert parse_question("Will there be 4 or more total cards shown?")["type"] == "total_cards"
    assert parse_question("Will a penalty kick be awarded OR a red card be shown?")["type"] == "penalty_or_red_card"
    assert parse_question("Will both teams score AND the match have 3 or more total goals?")["type"] == "btts_and_total_goals"


def test_poisson_helpers():
    from model.poisson import prob_over_under, prob_total_goals, halftime_outcome_probs, prob_x_greater_than_y
    # Over/under are complementary around the threshold boundary.
    p_over3 = prob_over_under(2.5, 3, "over")
    p_under2 = prob_over_under(2.5, 2, "under")
    assert abs((p_over3 + p_under2) - 1.0) < 1e-9
    # Total goals over 0 is essentially certain for a normal match.
    assert prob_total_goals(1.5, 1.2, 1, "over") > 0.9
    # Halftime probabilities sum to ~1 (small truncation residual allowed).
    h, t, a = halftime_outcome_probs(1.6, 1.0)
    assert abs((h + t + a) - 1.0) < 1e-4
    assert h > a  # stronger home side leads more often
    # Two equal Poissons: P(X>Y) is symmetric and below 0.5 (ties take mass).
    assert prob_x_greater_than_y(5.0, 5.0) < 0.5


def test_predict_btts():
    # If both teams have 0 xG, BTTS should be 0
    assert predict_btts(0, 0) == 0.0


def test_prob_btts_and_over():
    from model.poisson import prob_btts_and_over, predict_btts
    # BTTS-and-over is a subset of BTTS, so never larger than plain BTTS.
    for xh, xa in [(1.2, 0.9), (2.1, 0.8), (0.5, 0.5), (3.0, 2.4)]:
        for thr in (1, 2, 3, 4):
            joint = prob_btts_and_over(xh, xa, thr)
            assert 0.0 < joint < 1.0
            assert joint <= predict_btts(xh, xa) + 1e-9
    # Monotonic: higher threshold => lower-or-equal probability.
    for xh, xa in [(1.5, 1.1), (2.2, 1.7)]:
        vals = [prob_btts_and_over(xh, xa, t) for t in (1, 2, 3, 4)]
        for a, b in zip(vals, vals[1:]):
            assert a >= b - 1e-12


def test_elo_xg_adjustment_conserves_total():
    from model.elo import EloSystem, load_wc2026_ratings
    elo = EloSystem()
    for rh in (1000, 1500, 1800, 2200):
        for ra in (1000, 1500, 1800, 2200):
            for total in (0.5, 2.6, 4.0):
                h, a = elo.elo_xg_adjustment(rh, ra, total)
                assert abs((h + a) - total) <= 1e-9
                assert h > 0 and a > 0
    # Stronger team gets the larger share.
    h, a = elo.elo_xg_adjustment(2000, 1500, 2.6)
    assert h > a
    # Loader populates known teams and defaults unknowns to 1500.
    rated = load_wc2026_ratings()
    assert rated.get_rating("brazil") > 1500
    assert rated.get_rating("some unknown team") == 1500


def test_match_winner_uses_pure_market():
    # Match winner should equal the vig-free market prob exactly (no Poisson blend).
    index = build_odds_index(SAMPLE_ODDS)
    odds = find_match_odds(index, "Mexico vs South Africa")
    expected = format_prediction_for_submission(odds["p_home"])
    got = run_model_on_market(
        {"id": "mw", "question": "Will Mexico win the match?",
         "match": {"name": "Mexico vs South Africa"}}, index)
    assert got == expected


def test_team_score_half_fallback_above_50():
    # With no odds, half-scoring must fall back above 0.50 (>=55 here).
    pred = run_model_on_market(
        {"id": "h", "question": "Will Mexico score in the first half?",
         "match": {"name": "Unknownland vs Nowhere"}}, {})
    assert pred >= 50


def test_adjust_for_lineup():
    from bot.submit import adjust_for_lineup
    market = {"id": "p", "question": "Will Patrik Schick have at least 1 shot on target?",
              "match": {"name": "Czechia vs Ghana"}}
    starters = {"patrik schick", "tomas soucek"}
    # Starter -> unchanged.
    assert adjust_for_lineup(market, 60, starters) == 60
    # Benched (not in XI but lineup known) -> slashed.
    benched = {"tomas soucek", "vladimir coufal"}
    assert adjust_for_lineup(market, 60, benched) == max(1, round(60 * 0.30))
    # No lineup data -> unchanged.
    assert adjust_for_lineup(market, 60, None) == 60
    # Non-player market -> unchanged.
    mw = {"id": "x", "question": "Will Czechia win the match?",
          "match": {"name": "Czechia vs Ghana"}}
    assert adjust_for_lineup(mw, 55, benched) == 55


# Sample Odds API payload used by the wiring tests below.
SAMPLE_ODDS = [
    {
        "home_team": "Mexico",
        "away_team": "South Africa",
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Mexico", "price": 1.8},
                            {"name": "Draw", "price": 3.5},
                            {"name": "South Africa", "price": 4.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.9, "point": 2.5},
                            {"name": "Under", "price": 1.9, "point": 2.5},
                        ],
                    },
                ]
            }
        ],
    }
]


def test_normalize_and_split():
    # "Türkiye" resolves through the alias table to the canonical "turkey".
    assert normalize_team_name("Türkiye") == "turkey"
    assert normalize_team_name("Czechia") == "czech republic"
    assert normalize_team_name("GHA") == "ghana"
    assert normalize_team_name("  South Africa ") == "south africa"
    assert split_match_name("Mexico vs South Africa") == ("Mexico", "South Africa")


def test_odds_index_and_xg():
    index = build_odds_index(SAMPLE_ODDS)
    odds = find_match_odds(index, "Mexico vs South Africa")
    assert odds is not None
    # No-vig probabilities sum to 1.
    total = odds["p_home"] + odds["p_draw"] + odds["p_away"]
    assert abs(total - 1.0) < 1e-9
    # Mexico is the favorite, so its win prob is highest.
    assert odds["p_home"] > odds["p_away"]
    assert odds["total_goals"] == 2.5

    xg_home, xg_away = estimate_team_xg(
        odds["p_home"], odds["p_draw"], odds["p_away"], odds["total_goals"]
    )
    # Favorite gets more expected goals; total is close to the line.
    assert xg_home > xg_away
    assert abs((xg_home + xg_away) - 2.5) < 1e-9


def test_team_is_home():
    assert team_is_home("Mexico vs South Africa", "Mexico") is True
    assert team_is_home("Mexico vs South Africa", "South Africa") is False


def test_run_model_on_market_returns_valid_range():
    index = build_odds_index(SAMPLE_ODDS)
    markets = [
        {"id": "1", "question": "Will Mexico win the match?",
         "match": {"name": "Mexico vs South Africa"}},
        {"id": "2", "question": "Will the match have 3 or more total goals?",
         "match": {"name": "Mexico vs South Africa"}},
        {"id": "3", "question": "Will Mexico score in the first half?",
         "match": {"name": "Mexico vs South Africa"}},
        {"id": "4", "question": "Will Mexico have 4 or more shots on target?",
         "match": {"name": "Mexico vs South Africa"}},
        {"id": "5", "question": "At halftime, will the match be tied?",
         "match": {"name": "Mexico vs South Africa"}},
        {"id": "6", "question": "Will Uzbekistan have 5 or more corner kicks?",
         "match": {"name": "Mexico vs South Africa"}},
        {"id": "7", "question": "Will a penalty kick be awarded OR a red card be shown?",
         "match": {"name": "Mexico vs South Africa"}},
    ]
    for m in markets:
        pred = run_model_on_market(m, index)
        assert isinstance(pred, int)
        assert 1 <= pred <= 99


def test_favorite_gets_higher_win_prob():
    # Mexico is the favorite, so its modelled win prob should beat South Africa's.
    index = build_odds_index(SAMPLE_ODDS)
    mex = run_model_on_market(
        {"id": "a", "question": "Will Mexico win the match?",
         "match": {"name": "Mexico vs South Africa"}}, index)
    rsa = run_model_on_market(
        {"id": "b", "question": "Will South Africa win the match?",
         "match": {"name": "Mexico vs South Africa"}}, index)
    assert mex > rsa


def test_run_model_without_odds_uses_default():
    # No odds indexed -> match winner falls back to the 50 baseline.
    market = {"id": "x", "question": "Will Mexico win the match?",
              "match": {"name": "Mexico vs South Africa"}}
    pred = run_model_on_market(market, {})
    assert pred == 50


def test_player_goal_question_parses_as_player():
    # Regression: "score a goal (excluding own goals)" must route to the player
    # model, not team_score (which previously treated the player as a team).
    p = parse_question("Will Salem Al-Dawsari score a goal (excluding own goals)?")
    assert p["type"] == "player_goal_involvement"
    assert p["player"] == "Salem Al-Dawsari"
    # A plain team-scoring question must still parse as team_score.
    p2 = parse_question("Will Saudi Arabia score a goal?")
    assert p2["type"] == "team_score"
    assert p2["team"] == "Saudi Arabia"


def test_odds_alignment_when_feeds_flipped():
    # The Odds API lists Saudi Arabia as home, but SportsPredict names the match
    # "Spain vs Saudi Arabia" (Spain first). Spain's win prob must track Spain,
    # not be inverted to Saudi Arabia's.
    flipped = [{
        "home_team": "Saudi Arabia",
        "away_team": "Spain",
        "bookmakers": [{"markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Saudi Arabia", "price": 9.0},
                {"name": "Draw", "price": 5.0},
                {"name": "Spain", "price": 1.3},
            ]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.9, "point": 2.5},
            ]},
        ]}],
    }]
    index = build_odds_index(flipped)
    spain = run_model_on_market(
        {"id": "s", "question": "Will Spain win the match?",
         "match": {"name": "Spain vs Saudi Arabia"}}, index)
    saudi = run_model_on_market(
        {"id": "k", "question": "Will Saudi Arabia win the match?",
         "match": {"name": "Spain vs Saudi Arabia"}}, index)
    # Spain is the heavy favorite; its prob must be much higher than Saudi's.
    assert spain > 70
    assert spain > saudi


def test_btts_and_over_bounds_and_monotonic():
    from model.poisson import prob_btts_and_over, predict_btts
    btts = predict_btts(1.6, 1.2)
    p2 = prob_btts_and_over(1.6, 1.2, 2)
    p3 = prob_btts_and_over(1.6, 1.2, 3)
    p4 = prob_btts_and_over(1.6, 1.2, 4)
    # Each is a valid probability, <= plain BTTS, and decreasing in threshold.
    for p in (p2, p3, p4):
        assert 0.0 < p < 1.0
        assert p <= btts + 1e-9
    assert p2 >= p3 >= p4


def test_elo_xg_adjustment_conserves_total():
    from model.elo import load_wc2026_ratings
    elo = load_wc2026_ratings()
    h, a = elo.elo_xg_adjustment(elo.get_rating("brazil"), elo.get_rating("serbia"), 2.6)
    assert abs((h + a) - 2.6) < 1e-9
    assert h > a  # stronger side gets the larger share

def test_player_market_no_team_no_stats_is_conservative():
    # When the player's team can't be resolved (no squad cache) and there are no
    # real stats, the player market must fall back to a conservative prior near
    # 50 — never an extreme value like crediting a weak-team player at 60+.
    index = build_odds_index(SAMPLE_ODDS)
    pred = run_model_on_market(
        {"id": "p", "question": "Will Some Unknownplayer score a goal (excluding own goals)?",
         "match": {"name": "Mexico vs South Africa"}}, index)
    assert 25 <= pred <= 45  # shrunk base rate, not an extreme

if __name__ == "__main__":
    print("Running tests...")
    test_format_prediction()
    test_blend_probabilities()
    test_question_parser()
    test_poisson_helpers()
    test_predict_btts()
    test_prob_btts_and_over()
    test_elo_xg_adjustment_conserves_total()
    test_match_winner_uses_pure_market()
    test_team_score_half_fallback_above_50()
    test_adjust_for_lineup()
    test_normalize_and_split()
    test_odds_index_and_xg()
    test_team_is_home()
    test_run_model_on_market_returns_valid_range()
    test_favorite_gets_higher_win_prob()
    test_run_model_without_odds_uses_default()
    test_player_goal_question_parses_as_player()
    test_odds_alignment_when_feeds_flipped()
    test_btts_and_over_bounds_and_monotonic()
    test_elo_xg_adjustment_conserves_total()
    test_player_market_no_team_no_stats_is_conservative()
    print("All tests passed!")
