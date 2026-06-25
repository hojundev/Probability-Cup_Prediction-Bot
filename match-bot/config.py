"""
Match-bot configuration.

Tunable strategy parameters for the per-match prize bot. These are the only
knobs that differ from the competition-bot; everything else (model, data,
client) is reused unchanged from ../competition-bot.
"""

# Logit stretch factor applied to extremized markets. 1.0 = no change,
# 1.4-1.5 = recommended starting range, 2.0+ = very aggressive / high variance.
# 1.75 matches the aggressiveness of the strategy doc's example table
# (0.60 -> ~0.67, 0.70 -> ~0.79).
EXTREMIZE_K = 1.75

# Override of competition-bot's PERIPHERAL_SHRINK (0.35). For per-match prizes
# we keep 100% of the model's deviation from 0.50 instead of shrinking
# peripheral markets toward the coin flip. 1.0 = no shrinkage.
PERIPHERAL_SHRINK = 1.0

# Market types to push further from 0.50. These are the types where the model
# has real, directionally-reliable signal (odds-anchored or Poisson-based).
#
# Deliberately EXCLUDED (left at the competition-bot value):
#   - penalty_awarded / penalty_or_red_card: hardcoded base rates with no match
#     signal. Extremizing a constant is meaningless.
EXTREMIZE_TYPES = {
    "match_winner",
    "total_goals",
    "team_score",
    "team_score_half",
    "btts_and_total_goals",
    "halftime_tied",
    "halftime_winning",
    "team_more_than_opponent",
    "team_corners",
    "team_offsides",
    "team_cards",
    "total_cards",
    "total_sot",
    "team_total_sot",
    "player_shot_on_target",
    "player_goal_involvement",
}

# When set to a SportsPredict match name (e.g. "Ghana vs Panama"), the bot only
# submits predictions for that single match. None = submit for all open matches.
TARGET_MATCH = None

# Environment variable holding the second bot's SportsPredict API key. The
# competition allows 2 bots per account, each with its own key. This MUST be a
# different key from the competition-bot's SPORTSPREDICT_KEY.
API_KEY_ENV = "SPORTSPREDICT_KEY_BOT2"
