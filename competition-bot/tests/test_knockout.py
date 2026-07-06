"""
Round-of-32 (knockout) question-parser coverage.

Every question below is taken verbatim from the live R32 market dump. Each must
route to the intended type — in particular, NONE may fall through to "unknown"
(which submits a flat 0.50) and the knockout-specific mis-routes must be fixed
(player SOT, total corners/offsides, win-by-margin, both-teams-card, etc.).

Parser-only: imports nothing that pulls in scipy, so it runs anywhere.
Run: python3 tests/test_knockout.py   (or pytest tests/test_knockout.py)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.question_parser import parse_question

# (expected_type, question) — verbatim from the R32 dump.
CASES = [
    # ALG vs AUT
    ("team_more_than_opponent", "Will Algeria commit more fouls than Austria?"),
    ("team_more_than_opponent", "Will Austria have more shots on target than Algeria in the second half?"),
    ("team_first_goal", "Will Austria score the first goal of the second half?"),
    ("team_more_than_opponent", "Will Austria score more goals than Algeria in the second half?"),
    ("team_more_than_opponent", "Will Austria receive more cards than Algeria?"),
    ("match_winner", "Will Algeria win the match?"),
    ("total_corners", "Will there be 9 or more total corner kicks?"),
    ("team_total_sot", "Will Austria have 5 or more shots on target?"),
    ("player_shot_on_target", "Will Marcel Sabitzer have at least 1 shot on target?"),
    ("team_total_sot", "Will Algeria have 3 or more shots on target?"),

    # ARG vs CPV
    ("player_shot_on_target", "Will Lionel Messi (Argentina) have 3 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("goal_before_hydration", "Will a goal be scored before the first hydration break?"),
    ("player_goal_involvement", "Will Lautaro Martínez (Argentina) score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("team_clean_sheet", "Will Argentina keep a clean sheet in regulation (90 minutes + stoppage time)?"),
    ("match_winner", "Will Argentina win in regulation (90 minutes + stoppage time)?"),
    ("team_total_sot", "Will Cape Verde have 2 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("team_goals_over", "Will Argentina score 3 or more goals in regulation (90 minutes + stoppage time)?"),
    ("team_score_both_halves", "Will Argentina score in both halves in regulation (90 minutes + stoppage time)?"),
    ("team_total_sot", "Will Argentina have 8 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("player_goal_involvement", "Will Julián Álvarez (Argentina) score or assist a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("total_cards", "Will there be 3 or more total cards shown in regulation (90 minutes + stoppage time)?"),
    ("team_first_goal", "Will Argentina score the first goal of the match?"),
    ("team_corners", "Will Argentina have 8 or more corner kicks in regulation (90 minutes + stoppage time)?"),
    ("halftime_winning", "Will Argentina be ahead at halftime?"),
    ("total_shots", "Will there be 22 or more total shots (on and off target) in regulation (90 minutes + stoppage time)?"),

    # AUS vs EGY
    ("player_goal_involvement", "Will Mahmoud Trezeguet (Egypt) score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("halftime_tied", "Will the match be tied at halftime?"),
    ("total_goals", "Will the match have 2 or fewer total goals in regulation (90 minutes + stoppage time)?"),
    ("player_shot_on_target", "Will Nestory Irankunda (Australia) have 2 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("match_winner", "Will Egypt win in regulation (90 minutes + stoppage time)?"),
    ("both_teams_card", "Will both teams receive at least one card in regulation (90 minutes + stoppage time)?"),
    ("total_cards", "Will there be 4 or more total cards shown in regulation (90 minutes + stoppage time)?"),
    ("penalty_or_red_card", "Will a penalty kick be awarded OR a red card be shown in regulation (90 minutes + stoppage time)?"),
    ("team_total_sot", "Will Egypt have 5 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("total_corners", "Will there be 9 or more total corner kicks in regulation (90 minutes + stoppage time)?"),
    ("team_first_goal", "Will Egypt score the first goal of the match?"),
    ("half_vs_half_goals", "Will the second half produce more goals than the first half in regulation (90 minutes + stoppage time)?"),
    ("total_offsides", "Will there be 3 or more offside calls in regulation (90 minutes + stoppage time)?"),
    ("sub_before_half", "Will a substitution be made before halftime?"),
    ("total_shots", "Will there be 20 or more total shots (on and off target) in regulation (90 minutes + stoppage time)?"),

    # BRA vs JPN
    ("team_goals_over", "Will Brazil score 2 or more goals in regulation (90 minutes + stoppage time)?"),
    ("total_sot", "Will there be 8 or more total shots on target in regulation (90 minutes + stoppage time)?"),
    ("any_player_sot", "Will any player record 2 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("team_advance", "Will Japan advance to the Round of 16?"),
    ("total_corners", "Will there be 9 or more total corner kicks in regulation (90 minutes + stoppage time)?"),
    ("sub_scores", "Will a substitute score a goal in regulation (90 minutes + stoppage time)?"),
    ("player_goal_involvement", "Will Matheus Cunha score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("any_player_brace", "Will any player score more than 1 goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("team_offsides", "Will either team be ruled offside before the first hydration break?"),
    ("total_goals", "Will the match have 3 or more total goals in regulation (90 minutes + stoppage time)?"),
    ("team_score_both_halves", "Will Brazil score in both halves of regulation (90 minutes + stoppage time)?"),
    ("player_shot_on_target", "Will Ayase Ueda have at least 1 shot on target in regulation (90 minutes + stoppage time)?"),
    ("match_draw", "Will regulation (90 minutes + stoppage time) end in a tie?"),

    # --- New knockout question types ---
    ("total_goals_exact", "Will exactly 1 goal be scored in regulation (90 minutes + stoppage time)?"),
    ("total_goals_exact", "Will the match finish with exactly 2 total goals in regulation (90 minutes + stoppage time)?"),
    ("penalty_shootout", "Will the match be decided by a penalty shootout?"),

    # CIV vs NOR
    ("player_goal_involvement", "Will Erling Haaland (Norway) score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("team_more_than_opponent", "Will Ivory Coast have more corner kicks than Norway in regulation (90 minutes + stoppage time)?"),
    ("match_winner", "Will Norway win in regulation (90 minutes + stoppage time)?"),
    ("player_goal_involvement", "Will Martin Ødegaard (Norway) score or assist a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("btts", "Will both teams score in regulation (90 minutes + stoppage time)?"),
    ("player_shot_on_target", "Will Amad Diallo (Ivory Coast) have 1 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("player_shot_on_target", "Will Alexander Sørloth (Norway) have 2 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("team_first_goal", "Will Norway score the first goal of the match?"),
    ("total_offsides", "Will there be 4 or more offside calls in regulation (90 minutes + stoppage time)?"),
    ("penalty_awarded", "Will a penalty kick be awarded during regulation (90 minutes + stoppage time)?"),
    ("goal_after_hydration", "Will a goal be scored after the second hydration break in regulation (90 minutes + stoppage time)?"),

    # FRA vs SWE
    ("halftime_winning", "Will France be ahead at halftime?"),
    ("player_shot_on_target", "Will Kylian Mbappé (France) have 2 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("red_card", "Will a red card be shown in the match?"),
    ("team_corners", "Will France have 6 or more corner kicks in regulation (90 minutes + stoppage time)?"),
    ("match_winner", "Will France win in regulation (90 minutes + stoppage time)?"),
    ("player_shot_on_target", "Will Alexander Isak (Sweden) have 2 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("team_first_goal", "Will France score the first goal of the match?"),
    ("goal_first_half_stoppage", "Will a goal be scored in first-half stoppage (added) time?"),

    # GER vs PAR
    ("player_shot_on_target", "Will Jamal Musiala (Germany) have 2 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("team_first_goal", "Will Germany score the first goal of the match?"),
    ("player_shot_on_target", "Will Julio Enciso (Paraguay) have 1 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("team_score_both_halves", "Will Germany score in both halves in regulation (90 minutes + stoppage time)?"),
    ("team_corners", "Will Germany have 7 or more corner kicks in regulation (90 minutes + stoppage time)?"),
    ("halftime_winning", "Will Germany be ahead at halftime?"),

    # JOR vs ARG (group-style phrasing)
    ("team_corners", "Will Argentina have at least 1 corner kick in the first half?"),
    ("team_offsides", "Will Jordan be caught offside 2 or more times?"),
    ("team_score", "Will Jordan score at least 1 goal?"),
    ("team_score_half", "Will Jordan score in the second half?"),
    ("player_goal_involvement", "Will Mousa Al-Taamari score or assist a goal (excluding own goals)?"),

    # NED vs MAR
    ("halftime_winning", "Will Morocco be ahead at halftime?"),
    ("card_late", "Will a card be shown after the second hydration break, including any extra time?"),
    ("goal_first_half_stoppage", "Will a goal be scored in first-half stoppage time?"),
    ("goal_after_hydration", "Will a goal be scored after the second hydration break?"),
    ("half_vs_half_goals", "Will the second half produce more goals than the first half in regulation (90 minutes + stoppage time), excluding extra time?"),
    ("team_first_goal", "Will the Netherlands score the first goal of the match?"),
    # Brobbey is a Netherlands player, not a team -> player_shot_on_target.
    # (Previously fell to team_total_sot and relied on the submit.py safety net;
    # now routed directly since "N or more shots on target" + non-team subject.)
    ("player_shot_on_target", "Will Brian Brobbey have 2 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("team_more_than_opponent", "Will the Netherlands have more corner kicks than Morocco in regulation (90 minutes + stoppage time)?"),
    ("match_winner", "Will the Netherlands win in regulation (90 minutes + stoppage time)?"),
    ("btts", "Will both teams score in regulation (90 minutes + stoppage time)?"),
    ("half_total_goals", "Will the first half produce 2 or more goals?"),
    ("penalty_awarded", "Will a penalty kick be awarded during regulation (90 minutes + stoppage time)?"),

    # RSA vs CAN
    ("team_offsides", "Will either team be ruled offside before the first hydration break?"),
    ("team_advance", "Will South Africa advance to the Round of 16?"),
    ("halftime_winning", "Will Canada be ahead at halftime?"),
    ("player_shot_on_target", "Will Iqraam Rayners have at least 1 shot on target in regulation (90 minutes + stoppage time)?"),
    ("player_goal_involvement", "Will Cyle Larin score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),

    # USA vs BIH
    ("goal_second_half_stoppage", "Will a goal be scored in second-half stoppage time?"),
    ("team_corners", "Will 2 or more corner kicks be taken before the first hydration break?"),
    ("team_corners", "Will the United States have 6 or more corner kicks in regulation (90 minutes + stoppage time)?"),
    ("team_win_by_margin", "Will the United States win by 2 or more goals in regulation (90 minutes + stoppage time)?"),
    ("team_more_than_opponent", "Will Bosnia and Herzegovina receive more cards than the United States in regulation (90 minutes + stoppage time)?"),
    ("player_goal_involvement", "Will Folarin Balogun score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("card_first_half", "Will a card be shown in the first half?"),
    ("team_total_sot", "Will the United States have 6 or more shots on target in regulation (90 minutes + stoppage time)?"),
]


def test_no_unknowns():
    """No live R32 question should fall through to the 0.50 'unknown' prior."""
    bad = [(q, parse_question(q)["type"]) for _, q in CASES if parse_question(q)["type"] == "unknown"]
    assert not bad, "Unrecognised questions:\n" + "\n".join(f"  {q}" for q, _ in bad)


def test_routing():
    """Each question routes to the intended handler type."""
    mismatches = []
    for expected, q in CASES:
        got = parse_question(q)["type"]
        if got != expected:
            mismatches.append((expected, got, q))
    assert not mismatches, "Routing mismatches:\n" + "\n".join(
        f"  expected {e!r} got {g!r}: {q}" for e, g, q in mismatches
    )


def test_player_country_tag_stripped():
    """Player names keep accents but drop the '(Country)' tag for stat lookups."""
    p = parse_question("Will Lionel Messi (Argentina) have 3 or more shots on target?")
    assert p["type"] == "player_shot_on_target"
    assert p["player"] == "Lionel Messi"
    assert p["threshold"] == 3
    p2 = parse_question("Will Lautaro Martínez (Argentina) score a goal (excluding own goals)?")
    assert p2["player"] == "Lautaro Martínez"


def test_exact_goals_n_parsed():
    """total_goals_exact correctly extracts the exact goal count."""
    p1 = parse_question("Will exactly 1 goal be scored in regulation (90 minutes + stoppage time)?")
    assert p1["type"] == "total_goals_exact"
    assert p1["n"] == 1
    p2 = parse_question("Will the match finish with exactly 2 total goals in regulation (90 minutes + stoppage time)?")
    assert p2["type"] == "total_goals_exact"
    assert p2["n"] == 2
    p3 = parse_question("Will exactly 3 goals be scored in regulation (90 minutes + stoppage time)?")
    assert p3["type"] == "total_goals_exact"
    assert p3["n"] == 3


if __name__ == "__main__":
    test_no_unknowns()
    test_routing()
    test_player_country_tag_stripped()
    print(f"All {len(CASES)} knockout routing cases passed.")
