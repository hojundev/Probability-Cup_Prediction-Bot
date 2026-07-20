import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import match_bot
from match_bot import run_model_on_market
from extremize import extremize
from config import BINARY_TYPES, HEAVY_EXTREMIZE_K, HEAVY_EXTREMIZE_TYPES, LIGHT_EXTREMIZE_K

# Reuse the sample odds payload shape from the competition-bot tests.
from bot.match_data import build_odds_index

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


def _market(qid, question, name="Mexico vs South Africa"):
    return {"id": qid, "question": question, "match": {"name": name}}


def test_outputs_in_valid_range():
    index = build_odds_index(SAMPLE_ODDS)
    questions = [
        "Will Mexico win the match?",
        "Will the match have 3 or more total goals?",
        "Will Mexico score in the first half?",
        "Will Mexico have 4 or more shots on target?",
        "At halftime, will the match be tied?",
        "Will Mexico have 5 or more corner kicks?",
        "Will a penalty kick be awarded OR a red card be shown?",
    ]
    for i, q in enumerate(questions):
        pred = run_model_on_market(_market(str(i), q), index)
        assert isinstance(pred, int)
        assert 1 <= pred <= 99


def test_match_winner_is_binary():
    # match_winner is Tier 1 (BINARY_TYPES): the match-bot goes all-in to 1/99
    # in the direction the model favours.
    assert "match_winner" in BINARY_TYPES
    index = build_odds_index(SAMPLE_ODDS)
    from bot.submit import run_model_on_market as cb_run

    base = cb_run(_market("a", "Will Mexico win the match?"), index)
    out = run_model_on_market(_market("a", "Will Mexico win the match?"), index)
    assert out in (1, 99)
    assert out == (99 if base > 50 else 1)


def test_tier3_is_light_extremized():
    # penalty_or_red_card is in neither tier 1 nor tier 2 -> Tier 3 (light logit
    # stretch). Its raw base rate (0.24) is pushed further from 50.
    assert "penalty_or_red_card" not in BINARY_TYPES
    assert "penalty_or_red_card" not in HEAVY_EXTREMIZE_TYPES
    index = build_odds_index(SAMPLE_ODDS)
    from bot.submit import run_model_on_market as cb_run

    q = "Will a penalty kick be awarded OR a red card be shown?"
    base = cb_run(_market("p", q), index)     # 24, unshrunk base rate
    out = run_model_on_market(_market("p", q), index)
    assert out == round(min(99, max(1, 100 * extremize(base / 100, LIGHT_EXTREMIZE_K))))
    assert abs(out - 50) > abs(base - 50)


def test_player_goal_is_conservative():
    # Pure "score a goal" (player_goal) bypasses the tiers: submitted at the
    # model's calibrated probability, neither binary nor extremized.
    index = build_odds_index(SAMPLE_ODDS)
    from bot.submit import run_model_on_market as cb_run

    q = "Will Some Unknownplayer score a goal (excluding own goals)?"
    out = run_model_on_market(_market("g", q), index)
    assert out == cb_run(_market("g", q), index)


def test_peripheral_shrink_removed():
    # Corners are a peripheral type. The competition-bot shrinks them toward 50
    # (keep 0.35). The match-bot keeps full deviation AND extremizes, so its
    # prediction must be further from 50 than the competition-bot's.
    index = build_odds_index(SAMPLE_ODDS)
    from bot.submit import run_model_on_market as cb_run

    q = "Will Mexico have 5 or more corner kicks?"
    base = cb_run(_market("c", q), index)
    extreme = run_model_on_market(_market("c", q), index)
    # Only assert when the model isn't sitting exactly on 50.
    if base != 50:
        assert abs(extreme - 50) > abs(base - 50)


def test_target_match_filter():
    match_bot.TARGET_MATCH = "Ghana vs Panama"
    try:
        assert match_bot._target_match_matches("Ghana vs Panama") is True
        assert match_bot._target_match_matches("ghana vs panama") is True
        assert match_bot._target_match_matches("Mexico vs South Africa") is False
    finally:
        match_bot.TARGET_MATCH = None
    # None means match everything.
    assert match_bot._target_match_matches("anything at all") is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
