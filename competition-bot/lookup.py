"""
Match probability lookup tool.

Usage:
    python3 lookup.py TEAM1 TEAM2

Examples:
    python3 lookup.py GER CIV
    python3 lookup.py BRA Haiti
    python3 lookup.py TUR PAR
    python3 lookup.py New Zealand EGY

Team names must match how they appear in the API (run with no args to list all matches).
"""

import sys
import logging
logging.basicConfig(level=logging.WARNING)  # suppress info noise

# Disable api-football lookups for this read-only diagnostic tool.
# lookup.py should never write to the player cache — it's for inspection only.
import data.fetch_player_stats as _fps
_fps._api_disabled = True

from bot.client import get_probability_cup_lobby_and_event, fetch_matches, fetch_markets
from data.fetch_odds import fetch_market_odds
from bot.match_data import build_odds_index
from bot.submit import run_model_on_market

e, l = get_probability_cup_lobby_and_event()
matches = fetch_matches(e['id'], l['id'])

# No args — list all match names so you know what to type
if len(sys.argv) < 3:
    print("Available matches:")
    for m in sorted(matches, key=lambda x: x.get('name','')):
        print(f"  {m['name']}")
    print("\nUsage: python3 lookup.py TEAM1 TEAM2")
    sys.exit(0)

team1, team2 = sys.argv[1], sys.argv[2]

idx = build_odds_index(fetch_market_odds())
targets = [
    m for m in matches
    if team1.lower() in m.get('name','').lower()
    and team2.lower() in m.get('name','').lower()
]

if not targets:
    print(f"No match found containing '{team1}' and '{team2}'.")
    print("Run with no args to see all match names.")
    sys.exit(1)

for match in targets:
    print(f"\n=== {match['name']} ===")
    mk, _ = fetch_markets(l['id'], match['id'])
    print(f"  {'Question':<60} Prob")
    print("  " + "-" * 66)
    for m in mk:
        prob = run_model_on_market(m, idx)
        print(f"  {m['question']:<60} {prob}")
