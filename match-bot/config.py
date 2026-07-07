"""
Match-bot configuration — three-tier prediction strategy.

Tier 1 (BINARY_TYPES): submit 1 or 99. Only types where calibration shows the
model has been directionally right and RBP vs crowd has been positive.

Tier 2 (HEAVY_EXTREMIZE_TYPES): strong logit stretch (HEAVY_EXTREMIZE_K).
Types with real signal but less reliable direction than tier 1.

Tier 3 (everything else): light logit stretch (LIGHT_EXTREMIZE_K = 1.5).
Includes player markets and other types where the model over-predicts.
"""

# --- Tier 1: Binary (submit 1 or 99) ----------------------------------------
# Types where the model has been directionally correct and produced positive
# RBP vs crowd in the R32 recap. Going binary maximises gain/loss per market.
BINARY_TYPES = {
    "match_winner",         # +2.0 RBP in recap, directionally correct
    "team_advance",         # directionally correct
    "total_goals",          # under direction: +6.8 RBP in recap
    "total_corners",        # +4.3 RBP in recap
    "btts",                 # directionally consistent
    "team_score",           # directionally consistent
    "team_score_half",      # +1.3 RBP in recap
    "team_more_than_opponent",  # corner/foul comparisons: +5.1 RBP in recap
    "match_draw",           # driven by sharp market draw probability
    "team_win_by_margin",   # xG-derived, directionally sound
    "team_goals_over",      # xG-derived
    "team_clean_sheet",     # xG-derived
    "btts_and_total_goals", # +2.4 RBP in recap
}

# --- Tier 2: Heavy extremize -------------------------------------------------
# Genuine signal but less reliable direction. Push hard but not binary.
HEAVY_EXTREMIZE_K = 2.5
HEAVY_EXTREMIZE_TYPES = {
    "total_sot",            # under-predicted but directionally useful
    "total_shots",          # directionally useful
    "halftime_winning",     # genuine signal, moderate reliability
    "half_vs_half_goals",   # under-predicted, directionally right
    "team_first_goal",      # under-predicted massively (-29pts)
    "total_offsides",       # under-predicted but thin sample
    "halftime_tied",        # over-predicted but has signal
    "penalty_shootout",     # driven by draw probability
    "total_goals_exact",    # Poisson PMF, directional
    "team_corners",         # threshold version (comparative is tier 1)
    "team_offsides",        # xG-adjusted
    "team_total_sot",       # under-predicted; not binary yet
    "team_cards",           # xG-adjacent
    "both_teams_card",      # joint probability
    "goalkeeper_saves",     # SOT-derived
    "both_halves_same_goals",  # Poisson-derived
    "total_subs",           # flat base rate
    "team_score_both_halves",  # over-predicted but has structure
}

# --- Tier 3: Light extremize -------------------------------------------------
# Everything else: player markets (over-predicted), base-rate markets, and
# types with insufficient calibration data. k=1.5 = modest push from 50.
LIGHT_EXTREMIZE_K = 1.5
# (All types not in BINARY_TYPES or HEAVY_EXTREMIZE_TYPES land here)

# Peripheral shrink override — no shrink for the match-bot (1.0 = keep full
# deviation). Tier 1 bypasses this anyway since it goes binary.
PERIPHERAL_SHRINK = 1.0

# When set to a SportsPredict match name (e.g. "Ghana vs Panama"), the bot only
# submits predictions for that single match. None = submit for all open matches.
TARGET_MATCH = None

# Environment variable holding the second bot's SportsPredict API key.
API_KEY_ENV = "SPORTSPREDICT_KEY_BOT2"
