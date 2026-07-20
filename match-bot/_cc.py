import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match_bot as mb
from bot.match_data import build_odds_index
from bot.submit import reset_player_budget
from data.fetch_odds import fetch_market_odds
idx = build_odds_index(fetch_market_odds()); reset_player_budget()
qs = [
    ("player_goal (Messi)",   "Will Lionel Messi (Argentina, #10) score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("player_goal (Kane)",    "Will Harry Kane (England, #9) score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("goal_involvement",      "Will Jude Bellingham (England, #10) score or assist a goal (excluding own goals) in regulation (90 minutes + stoppage time)?"),
    ("shot_on_target",        "Will Julian Alvarez (Argentina, #9) have 1 or more shots on target in regulation (90 minutes + stoppage time)?"),
    ("match_winner",          "Will England win in regulation (90 minutes + stoppage time)?"),
]
for label, q in qs:
    p = mb.run_model_on_market({"id":"x","question":q,"match":{"name":"England vs Argentina"}}, idx)
    print(f"{p:>3}  [{label}]")
