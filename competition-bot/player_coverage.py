"""
Player-stats coverage report.

Lists every player named in the current open markets and whether the player has
real hand-curated stats, a cached fallback (api-football miss — won't be
retried), or no cache entry yet. Use it to see exactly who to add real per-90
stats for in data/.player_cache.json.

Usage:
    python3 player_coverage.py            # all matches
    python3 player_coverage.py BRA        # only matches whose name contains BRA

Classification:
    REAL      -> is_real: true   (your manual entry; priced on real per-90 stats)
    FALLBACK  -> is_real: false  (cached api-football miss; priced off team xG,
                                  and NEVER re-fetched — overwrite to fix)
    MISSING   -> not cached yet  (will trigger one live lookup next run, then
                                  almost certainly cache as a FALLBACK)
"""

import sys
import logging
from collections import defaultdict

logging.basicConfig(level=logging.WARNING)  # suppress info noise

from bot.client import get_probability_cup_lobby_and_event, fetch_matches, fetch_markets
from bot.question_parser import parse_question
from bot.match_data import split_match_name, normalize_team_name
from data.fetch_player_stats import peek_cache

# Question types whose subject is a single player.
_PLAYER_TYPES = {"player_shot_on_target", "player_goal_involvement"}


def _match_sides(match_name):
    home, away = split_match_name(match_name)
    return {normalize_team_name(home), normalize_team_name(away)}


def _player_in_market(parsed, match_name):
    """Return the player's name for a player market, else None."""
    qtype = parsed.get("type")
    if qtype in _PLAYER_TYPES:
        return parsed.get("player")
    # Knockout player SOT without a "(Country)" tag parses as team_total_sot;
    # if the "team" isn't one of the two sides, it's really a player.
    if qtype == "team_total_sot":
        team = parsed.get("team", "")
        if team and team != "Unknown" and normalize_team_name(team) not in _match_sides(match_name):
            return team
    return None


def _classify(name):
    entry = peek_cache(name)
    if entry is None:
        return "MISSING"
    return "REAL" if entry.get("is_real") else "FALLBACK"


def main():
    name_filter = " ".join(sys.argv[1:]).strip().lower()

    event, lobby = get_probability_cup_lobby_and_event()
    matches = fetch_matches(event["id"], lobby["id"])
    if name_filter:
        matches = [m for m in matches if name_filter in m.get("name", "").lower()]

    # player name -> set of match names they appear in
    players = defaultdict(set)
    for match in matches:
        match_name = match.get("name", "")
        markets, _ = fetch_markets(lobby["id"], match["id"])
        for mk in markets:
            parsed = parse_question(mk.get("question", ""))
            player = _player_in_market(parsed, match_name)
            if player:
                players[player].add(match_name)

    buckets = {"REAL": [], "FALLBACK": [], "MISSING": []}
    for name in players:
        buckets[_classify(name)].append(name)

    for label in ("REAL", "FALLBACK", "MISSING"):
        names = sorted(buckets[label])
        print(f"\n{label} ({len(names)})")
        print("-" * 50)
        for n in names:
            where = ", ".join(sorted(players[n]))
            print(f"  {n:<28} {where}")

    total = sum(len(v) for v in buckets.values())
    print("\n" + "=" * 50)
    print(f"{total} distinct players in open markets: "
          f"{len(buckets['REAL'])} real, "
          f"{len(buckets['FALLBACK'])} fallback, "
          f"{len(buckets['MISSING'])} missing")
    print("Add real per-90 stats (is_real: true) for FALLBACK/MISSING players "
          "you care about in data/.player_cache.json.")


if __name__ == "__main__":
    main()
