import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import match_bot
from match_bot import run_model_on_market
from extremize import extremize
from config import EXTREMIZE_K

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


def test_match_winner_is_extremized():
    # match_winner is in EXTREMIZE_TYPES, so the match-bot's prediction must be
    # further from 50 than the competition-bot's raw market line.
    index = build_odds_index(SAMPLE_ODDS)
    from bot.submit import run_model_on_market as cb_run

    base = cb_run(_market("a", "Will Mexico win the match?"), index)
    extreme = run_model_on_market(_market("a", "Will Mexico win the match?"), index)
    assert abs(extreme - 50) > abs(base - 50)
    # And it should match the extremize transform of the base decimal.
    assert extreme == round(min(99, max(1, 100 * extremize(base / 100, EXTREMIZE_K))))


def test_penalty_market_not_extremized():
    # penalty_or_red_card is NOT in EXTREMIZE_TYPES: hardcoded base rate, the
    # match-bot must leave it identical to the competition-bot.
    index = build_odds_index(SAMPLE_ODDS)
    from bot.submit import run_model_on_market as cb_run

    q = "Will a penalty kick be awarded OR a red card be shown?"
    assert run_model_on_market(_market("p", q), index) == cb_run(_market("p", q), index)


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
