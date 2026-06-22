"""
Print all match-bot predictions for a given match name.

Usage:
    python check_submissions.py "Argentina vs Austria"

Uses SPORTSPREDICT_KEY_BOT2 from .env so you see what the match-bot submitted,
not the competition-bot.
"""

import os
import sys

# Make competition-bot importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMPETITION_BOT = os.path.abspath(os.path.join(_HERE, "..", "competition-bot"))
if _COMPETITION_BOT not in sys.path:
    sys.path.insert(0, _COMPETITION_BOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))

import bot.client as client_module
from bot.client import SportsPredictClient, get_probability_cup_lobby_and_event, fetch_markets, fetch_matches, fetch_my_predictions, join_lobby

# Point at the match-bot's key.
key = os.getenv("SPORTSPREDICT_KEY_BOT2")
if not key:
    print("SPORTSPREDICT_KEY_BOT2 not set in .env")
    sys.exit(1)
client_module._default_client = SportsPredictClient(api_key=key)

target = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
if not target:
    print("Usage: python check_submissions.py \"Argentina vs Austria\"")
    sys.exit(1)

event, lobby = get_probability_cup_lobby_and_event()
lobby_id = lobby["id"]
join_lobby(lobby_id)

matches = fetch_matches(event["id"], lobby_id)
match = next(
    (m for m in matches if target.lower() in m.get("name", "").lower()),
    None,
)
if not match:
    available = [m.get("name") for m in matches]
    print(f"Match '{target}' not found. Available matches:")
    for name in available:
        print(f"  {name}")
    sys.exit(1)

markets, _ = fetch_markets(lobby_id, match["id"])
market_map = {m["id"]: m for m in markets}

predictions = fetch_my_predictions(lobby_id)
# Filter to this match's markets.
preds = [p for p in predictions if p["market_id"] in market_map]

if not preds:
    print(f"No predictions found for '{match['name']}'.")
    sys.exit(0)

print(f"\nMatch-bot predictions for: {match['name']}")
print(f"{'Question':<70} {'Prob':>5}")
print("-" * 77)
for p in sorted(preds, key=lambda x: market_map[x["market_id"]].get("question", "")):
    question = market_map[p["market_id"]].get("question", p["market_id"])
    prob = p.get("probability", "?")
    print(f"{question:<70} {prob:>4}%")
