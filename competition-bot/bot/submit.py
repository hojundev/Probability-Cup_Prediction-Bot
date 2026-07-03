import logging
from datetime import datetime, timedelta, timezone

from bot.client import (
    get_probability_cup_lobby_and_event,
    fetch_matches,
    fetch_markets,
    fetch_my_predictions,
    submit_predictions_batch,
    update_prediction,
    join_lobby,
)
from bot.question_parser import parse_question
from bot.match_data import (
    build_odds_index,
    find_match_odds,
    estimate_team_xg,
    team_is_home,
    split_match_name,
    normalize_team_name,
)
from data.fetch_odds import fetch_market_odds
from data.fetch_player_stats import fetch_player_stats, peek_cache
from model.poisson import (
    predict_btts,
    prob_at_least_one,
    prob_over_under,
    prob_total_goals,
    prob_btts_and_over,
    halftime_outcome_probs,
    prob_x_greater_than_y,
    prob_win_by_margin,
)
from model.ensemble import blend_probabilities, format_prediction_for_submission
from model.elo import load_wc2026_ratings

logger = logging.getLogger(__name__)

# Share of goals scored in each half (World Cup historical split).
FIRST_HALF_GOAL_SHARE = 0.43
SECOND_HALF_GOAL_SHARE = 0.57

# Convert a team's expected goals into expected shots on target.
# The ratio scales with team xG: dominant favorites take disproportionately more
# shots per xG (lots of speculative attempts) than average/weak teams. So the
# effective ratio rises above/below the base by how far the team's xG sits above
# an average team, clamped to a sane range.
#
#   sot_per_xg(team_xg) = SOT_PER_XG_BASE * (1 + SOT_PER_XG_SLOPE * (team_xg/AVG_TEAM_XG - 1))
#
# At team_xg = AVG_TEAM_XG the ratio is exactly SOT_PER_XG_BASE (3.3, calibrated
# from the ~4.3-SOT / ~1.3-xG match average).
SOT_PER_XG = 3.3               # base ratio (kept for fallbacks that lack a team xG)
SOT_PER_XG_BASE = 3.3
SOT_PER_XG_SLOPE = 0.35        # how strongly the ratio rises with team dominance
SOT_PER_XG_MIN = 2.9           # clamp: weak teams don't drop below this
SOT_PER_XG_MAX = 4.2           # clamp: elite favorites don't exceed this
MIN_TEAM_SOT = 2.5

# Historical per-match base rates used as calibrated priors for peripheral
# markets the bookmaker feed doesn't cover. These beat a naive 0.50.
AVG_CORNERS_PER_TEAM = 5.5      # per team; calibrated up (team-corner overs were under-predicted)
AVG_TOTAL_CORNERS = 9.0         # both teams; calibrated down (total-corner overs were over-predicted)
AVG_OFFSIDES_PER_TEAM = 1.7
AVG_CARDS_TOTAL = 2.665        # yellow + red per match; 2026 WC actuals: 2.54Y + 0.125R

# --- xG-adjusted offsides ---------------------------------------------------
# A flat offside rate ignores playing style: attacking sides (high xG) push the
# last line and get caught offside more often, while defensive sides (low xG)
# get caught less. We scale the base offside rate by the team's xG relative to
# an average team, blending so extreme xG values don't over-react.
#
#   xg_ratio = team_xg / AVG_TEAM_XG
#   mu = AVG_OFFSIDES_PER_TEAM * (1 + OFFSIDE_XG_SCALE * (xg_ratio - 1))
#
# OFFSIDE_XG_SCALE = 0 ignores xG (flat fallback); 1 = full proportional scaling.
AVG_TEAM_XG = 1.3              # average xG per team per match
OFFSIDE_XG_SCALE = 0.6        # how strongly xG modulates the offside rate
CORNER_XG_SCALE = 0.5         # how strongly xG modulates corner counts (attacking sides win more)
AVG_FOULS_PER_TEAM = 12.0
PENALTY_AWARDED_RATE = 0.26    # P(>=1 penalty in a match)
PENALTY_OR_RED_RATE = 0.29     # P(penalty OR red card); calibrated down from group-stage results

TEAM_SHOTS_PER_GAME = 13.0
TEAM_SHOTS_ON_TARGET_PER_GAME = 4.5

# Weight on the (sharp) betting-market line when blending with the model.
# Sharp books are sharper than the forecasting crowd, so on odds-anchored
# markets we ride the market hard and add only a little model.
MARKET_ALPHA = 0.88
DEFAULT_PROB = 0.50

# --- Elo blending -----------------------------------------------------------
# Fraction of the final xG that comes from the (independent) Elo signal; the
# rest comes from the market-derived xG. Small, because the market is sharper —
# Elo is here to add a little independent information, not to dominate.
ELO_BLEND_WEIGHT = 0.15
ELO = load_wc2026_ratings()

# --- Player fallbacks (used when live per-90 stats are unavailable) ---------
# Roughly how many outfield players generate the team's shots on target, and
# how many share its expected goals. Used to turn a team-level xG into a
# sensible single-player prior when api-football has no data for the player.
TEAM_AVG_PLAYERS_SHOOTING = 4.0
TEAM_AVG_GOAL_SCORERS = 2.5
PLAYER_ASSIST_RATE = 0.15      # share of team xG attributable to one player's assists
MIN_REAL_STAT = 0.01           # below this a "real" per-90 stat is treated as a miss
# Cap live api-football player lookups per run to protect the 100/day quota.
MAX_PLAYER_REQUESTS_PER_RUN = 20

# Base rates used when a player's team can't be identified AND we have no real
# stats — a deliberately conservative prior (shrunk toward 50) so we never post
# wildly wrong numbers like crediting a weak-team player with elite scoring.
PLAYER_GOAL_INVOLVEMENT_BASE = 0.30   # P(goal or assist) for a generic featured player
PLAYER_SOT_BASE = 0.45                # P(>=1 shot on target) for a generic featured player

# --- Player market scaling --------------------------------------------------
# A player's probability is their OWN per-90 rate (or a team-xG estimate when we
# have no stats), adjusted ONLY by how strong their team is expected to be in
# this match. No pull toward a generic crowd baseline.
#
# The adjustment is a single multiplier on the rate:
#
#   context = clamp(team_xg / PLAYER_TEAM_XG_REF, PLAYER_TEAM_XG_FLOOR, PLAYER_TEAM_XG_CEIL)
#   player_rate = own_per90_rate * context
#
# PLAYER_TEAM_XG_REF is the team xG at which a player realizes ~their full rate;
# weaker (underdog) teams scale their players down proportionally, stronger
# teams sit near full. When the team/match xG can't be resolved (no odds, or the
# player isn't in the squad cache) we assume an average team (AVG_TEAM_XG).
#
# Tuning:
#   - PLAYER_TEAM_XG_REF: raise to make EVERY player more conservative (only the
#     strongest teams' players approach their raw rate); lower to trust club
#     rates more.
#   - PLAYER_TEAM_XG_CEIL: max multiplier (1.0 = a strong-team player tops out at
#     their raw rate, no upside boost).
PLAYER_TEAM_XG_REF = 1.9      # team xG corresponding to ~full per-90 rate realization
PLAYER_TEAM_XG_FLOOR = 0.20   # min context multiplier (don't zero out a star on a minnow)
PLAYER_TEAM_XG_CEIL = 1.00    # max context multiplier (1.0 = no upside boost)

# Player markets for absent/benched players are slashed to this fraction of the
# starter probability once confirmed lineups are in (see lineup updates).
BENCH_PLAYER_FACTOR = 0.30

# --- Knockout / Round-of-32 market parameters -------------------------------
# Knockout rounds add markets the group stage never had. The ones below are
# either derived from xG (handled in the model) or priced from calibrated base
# rates when no signal exists. These base rates are deliberately NOT 0.50 and
# are NOT shrunk toward 50 (same reasoning as the penalty markets) — shrinking a
# confident base rate would corrupt it. Tune as real knockout data accrues.
RED_CARD_RATE = 0.118             # P(>=1 red card); derived from 2026 WC avg 0.125 R/match
SUB_SCORES_RATE = 0.18            # P(a substitute scores)
SUB_BEFORE_HALF_RATE = 0.16       # P(a substitution before halftime; usually injury)
ANY_PLAYER_BRACE_RATE = 0.20      # P(some player scores 2+ goals)
ANY_PLAYER_MULTI_SOT_RATE = 0.88  # P(some player records 2+ shots on target) — very common
CARD_LATE_RATE = 0.65             # P(card after 2nd hydration break / late, incl. ET)
CARD_FIRST_HALF_SHARE = 0.35      # share of a match's cards shown in the first half (WC: ~35%)

# Total shots (on + off target) in an average match; scaled by the match's
# combined xG relative to an average game.
TOTAL_SHOTS_BASELINE = 2 * TEAM_SHOTS_PER_GAME   # ~26 shots in an average match

# Fraction of a match's total goals expected inside the specific time windows
# the knockout markets ask about. Based on actual average break times:
#   First hydration break:  ~22'  (window = 0' to 22')
#   Second hydration break: ~67'  (window = 67' to 90'+stoppage ~5' = ~28 min)
#
# Goals are NOT uniformly distributed across 90 minutes:
#   - 0-22' (before 1st break): ~20% of goals — teams settle in, defences set
#   - 67-90'+5' (after 2nd break): ~38% of goals — fatigue, subs, trailing teams
#     pushing forward, stoppage-time goals (~5' window punches way above its share)
#
# Calibrated against group-stage results: before-hydration was slightly
# over-predicted, after-hydration was meaningfully under-predicted.
GOAL_FRAC_BEFORE_HYDRATION = 0.20        # before the ~22' first hydration break
GOAL_FRAC_AFTER_HYDRATION = 0.38         # after the ~67' second hydration break
GOAL_FRAC_FIRST_HALF_STOPPAGE = 0.05     # first-half added time
GOAL_FRAC_SECOND_HALF_STOPPAGE = 0.09    # second-half added time
# Fallback P(goal in window) when no odds/xG are available.
GOAL_WINDOW_FALLBACK = {
    "before_hydration": 0.34,
    "after_hydration": 0.62,
    "first_half_stoppage": 0.10,
    "second_half_stoppage": 0.16,
}

# --- Lineup-update polling window -------------------------------------------
# Start checking for confirmed lineups this many minutes before kickoff, and
# poll on this interval until kickoff. Lineups are typically released ~60 min
# out; a 90-minute window with 15-minute polling catches them shortly after.
LINEUP_WINDOW_MINUTES = 90
LINEUP_CHECK_INTERVAL_MINUTES = 15

# --- Confidence shrinkage for signal-less peripheral markets ----------------
# We can't see the crowd before submitting, so on markets where the model has
# weak/no real edge we shrink predictions toward 50 to avoid confident
# wrong-side bets, which the Brier rule punishes hard. SHRINK is the fraction
# of the model's deviation from 0.50 that we KEEP (0 = always 50, 1 = none).
#
# Only markets that are genuinely near coin-flip with a WEAK heuristic lean go
# here. Excluded on purpose:
#   - penalty_awarded / penalty_or_red_card: calibrated base rates that sit far
#     from 50 for real reasons (0.26 / 0.40) — shrinking would corrupt them.
#   - team_first_goal: derived from market xG (a real anchored signal).
#   - scoring / totals / match_winner: odds-anchored, keep full sharpness.
PERIPHERAL_SHRINK = 0.35
PERIPHERAL_TYPES = {
    "team_corners",
    "team_offsides",
    "team_cards",
    "total_cards",
    "total_corners",
    "total_offsides",
    "team_more_than_opponent",
}

# Per-type shrink overrides (fraction of deviation from 50 KEPT). Group-stage
# calibration: team_corners was UNDER-predicted (hugging 50 from below), so it
# keeps more of its now-xG-scaled deviation. total_corners was OVER-predicted,
# so it keeps the default heavier shrink — lightening it would amplify the over.
PERIPHERAL_SHRINK_OVERRIDES = {
    "team_corners": 0.65,
}


def _shrink_to_half(prob, keep):
    """Pull a probability toward 0.50, keeping `keep` of its deviation."""
    return 0.50 + (prob - 0.50) * keep


def _sot_per_xg(team_xg):
    """
    Shots-on-target-per-xG ratio for a team, scaled by its attacking dominance.
    Favorites (high xG) convert xG into more shots on target than average/weak
    teams. Falls back to the base ratio when team_xg is unavailable.
    """
    if not team_xg:
        return SOT_PER_XG_BASE
    ratio = SOT_PER_XG_BASE * (1 + SOT_PER_XG_SLOPE * (team_xg / AVG_TEAM_XG - 1))
    return max(SOT_PER_XG_MIN, min(SOT_PER_XG_MAX, ratio))


def _match_name(market):
    match = market.get("match") or {}
    return match.get("name", "")


def _half_share(half):
    if half == "first":
        return FIRST_HALF_GOAL_SHARE
    if half == "second":
        return SECOND_HALF_GOAL_SHARE
    return 1.0


def _team_xg_for(parsed, match_name, xg_home, xg_away):
    """Pick the xG belonging to the team named in the question."""
    if xg_home is None:
        return None
    team = parsed.get("team", "")
    return xg_home if team_is_home(match_name, team) else xg_away


def _team_side(match_name, team):
    """
    Return 'home', 'away', or None for `team` within `match_name`.

    None means the name matches neither side — typically because it's actually a
    player name (e.g. a knockout player market that lacked a "(Country)" tag and
    fell into a team handler), which callers use to reroute appropriately.
    """
    home, away = split_match_name(match_name)
    t = normalize_team_name(team)
    if not t:
        return None
    if t == normalize_team_name(home):
        return "home"
    if t == normalize_team_name(away):
        return "away"
    return None


def _player_team_xg(player_name, match_name, xg_home, xg_away):
    """
    Return the xG of the player's actual national team, resolved from the cached
    squad map, or None if the player's team can't be identified (or no xG).

    The player's team isn't in the question text, so this is how we avoid
    crediting a weak-team player with the opponent's (or the average) xG.
    """
    from data.fetch_squads import resolve_player_team

    if xg_home is None:
        return None
    team = resolve_player_team(player_name)
    if not team:
        return None
    home, away = split_match_name(match_name)
    if team == normalize_team_name(home):
        return xg_home
    if team == normalize_team_name(away):
        return xg_away
    return None


def _aligned_odds(odds, match_name):
    """
    Return (p_home, p_draw, p_away, total_goals) aligned to the SportsPredict
    match-name ordering.

    The Odds API picks its own home/away designation, which for neutral-venue
    World Cup games need not match SportsPredict's "TeamA vs TeamB" ordering.
    We key the odds index by an unordered team pair, so `p_home`/`p_away` in the
    stored dict follow the *odds feed* ordering. Here we re-map them to the
    SportsPredict home team so all downstream `team_is_home` checks are correct.
    """
    sp_home, _ = split_match_name(match_name)
    odds_away_norm = normalize_team_name(odds.get("away_team", ""))
    # If the odds feed's AWAY team is actually SportsPredict's HOME team, the
    # two feeds are flipped relative to each other — swap home/away.
    if odds_away_norm and odds_away_norm == normalize_team_name(sp_home):
        return odds["p_away"], odds["p_draw"], odds["p_home"], odds.get("total_goals")
    return odds["p_home"], odds["p_draw"], odds["p_away"], odds.get("total_goals")


# --- Per-run player-stats budget -------------------------------------------
# Tracks unique player names that triggered (or would trigger) a live lookup
# this run. Once the cap is hit, further player markets fall back to xG-derived
# priors instead of consuming api-football quota.
_player_request_names = set()


def reset_player_budget():
    """Clear the per-run player-lookup budget. Call at the start of each run."""
    _player_request_names.clear()


def _real_player_stats(player_name):
    """
    Return real per-90 stats for a player, or None when we should use an
    xG-based fallback instead. Returns None when:
      - the name is missing/unknown,
      - the player needs a live lookup but the per-run NETWORK budget is
        exhausted (cached players are always served, free of budget),
      - api-football has no real data (is_real is False), or
      - the stats are present but effectively zero (below MIN_REAL_STAT).

    The per-run budget only throttles actual api-football calls (cache misses),
    NOT cache hits — so manually-curated entries are always used regardless of
    how many player markets a run contains.
    """
    if not player_name or player_name == "Unknown":
        return None

    cached = peek_cache(player_name)
    if cached is None:
        # Not cached -> a live api-football call is required; spend from the
        # per-run network budget (and skip once it's exhausted).
        key = player_name.strip().lower()
        if key not in _player_request_names:
            if len(_player_request_names) >= MAX_PLAYER_REQUESTS_PER_RUN:
                return None
            _player_request_names.add(key)
        stats = fetch_player_stats(player_name)
    else:
        # Cache hit (manual real entry or a previously-cached miss): free.
        stats = cached

    if not stats or not stats.get("is_real"):
        return None
    return stats


def _player_team_context(player, match_name, xg_home, xg_away):
    """
    The SOLE adjustment to a player's rate: how strong their team is expected to
    be in this match, as team_xg / PLAYER_TEAM_XG_REF (clamped). Weak/underdog
    teams scale their players down; strong teams sit near their full rate. When
    the team or match xG can't be resolved (no odds, or player not in the squad
    cache) we assume an average team.
    """
    team_xg = _player_team_xg(player, match_name, xg_home, xg_away)
    if team_xg is None:
        team_xg = AVG_TEAM_XG
    return max(PLAYER_TEAM_XG_FLOOR, min(PLAYER_TEAM_XG_CEIL, team_xg / PLAYER_TEAM_XG_REF))


def _player_sot_prob(player, threshold, half, match_name, xg_home, xg_away):
    """
    Probability a single player records `threshold`+ shots on target.

    The player's own SoT/90 (or a team-xG estimate when we have no stats) is
    adjusted ONLY by their team's expected performance in this match — no pull
    toward a crowd baseline. Shared by the `player_shot_on_target` handler and
    the `team_total_sot` safety net (knockout player markets parsed as a team).
    """
    threshold = threshold or 1
    context = _player_team_context(player, match_name, xg_home, xg_away)
    stats = _real_player_stats(player)
    if stats and stats.get("shots_on_target_per_90", 0.0) >= MIN_REAL_STAT:
        # Real stats: player's own rate, scaled by team performance.
        player_xsot = stats["shots_on_target_per_90"] * context
    else:
        team_xg = _player_team_xg(player, match_name, xg_home, xg_away)
        if team_xg is None:
            # No stats and no team/odds -> no information; conservative default.
            base = PLAYER_SOT_BASE * (0.5 ** (threshold - 1))
            if half:
                base *= _half_share(half)
            return base
        # Team known but no player stats: team-xG-derived per-player estimate
        # (already reflects team performance, so no extra context multiplier).
        player_xsot = (team_xg * _sot_per_xg(team_xg)) / TEAM_AVG_PLAYERS_SHOOTING
    if half:
        player_xsot *= _half_share(half)
    if threshold <= 1:
        return prob_at_least_one(player_xsot)
    return prob_over_under(player_xsot, threshold, "over")


def _model_prob_for_market(market, odds_index):
    """Route a market to the right model and return a decimal probability (0-1)."""
    question = market.get("question", "")
    parsed = parse_question(question)
    qtype = parsed["type"]
    match_name = _match_name(market)

    odds = find_match_odds(odds_index, match_name) if match_name else None
    xg_home = xg_away = None
    total_xg = None
    p_home_sp = p_draw_sp = p_away_sp = None
    if odds:
        # Align the bookmaker home/away to the SportsPredict match ordering so
        # every team_is_home check downstream lines up with these values.
        p_home_sp, p_draw_sp, p_away_sp, total_goals = _aligned_odds(odds, match_name)
        xg_home, xg_away = estimate_team_xg(
            p_home_sp, p_draw_sp, p_away_sp, total_goals
        )
        # Blend in an independent Elo-derived xG split so the model is not a pure
        # restatement of the bookmaker line. Conserves total xG.
        home, away = split_match_name(match_name)
        rh = ELO.get_rating(normalize_team_name(home))
        ra = ELO.get_rating(normalize_team_name(away))
        elo_h, elo_a = ELO.elo_xg_adjustment(rh, ra, xg_home + xg_away)
        xg_home = (1 - ELO_BLEND_WEIGHT) * xg_home + ELO_BLEND_WEIGHT * elo_h
        xg_away = (1 - ELO_BLEND_WEIGHT) * xg_away + ELO_BLEND_WEIGHT * elo_a
        total_xg = xg_home + xg_away

    # ---------- Match winner ----------
    # Pure market: the sharp vig-free line is better than anything our Poisson
    # model can re-derive from it, so we return it directly (no Poisson call).
    if qtype == "match_winner":
        team = parsed.get("team", "")
        if odds:
            is_home = team_is_home(match_name, team)
            return p_home_sp if is_home else p_away_sp
        return DEFAULT_PROB

    # ---------- Team scores (full match) ----------
    if qtype == "team_score":
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        return prob_at_least_one(team_xg) if team_xg else 0.62  # base rate a team scores

    # ---------- Team scores in a half ----------
    if qtype == "team_score_half":
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        if team_xg is None:
            # Empirical WC base rate that a team scores in a given half is ~0.55,
            # comfortably above a coin flip — never fall back below 0.50.
            return 0.55
        return prob_at_least_one(team_xg * _half_share(parsed.get("half")))

    # ---------- Team scores the first goal of (game/half) ----------
    if qtype == "team_first_goal":
        # Roughly: P(team scores first) ~ share of combined scoring rate.
        if total_xg:
            team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
            half = parsed.get("half")
            if half:
                # P(team scores first in the half) ~ team rate / combined,
                # times P(any goal in the half).
                share = _half_share(half)
                mu_team = team_xg * share
                mu_total = total_xg * share
                p_any = prob_at_least_one(mu_total)
                return p_any * (mu_team / mu_total) if mu_total > 0 else DEFAULT_PROB
            # Whole game: share of combined scoring rate, scaled by the chance
            # that any goal is scored at all (avoids overestimating on 0-0).
            return (team_xg / total_xg) * prob_at_least_one(total_xg) if total_xg > 0 else DEFAULT_PROB
        return 0.30

    # ---------- Total goals (match) ----------
    if qtype == "total_goals":
        if total_xg is not None:
            return prob_total_goals(xg_home, xg_away, parsed["threshold"], parsed["direction"])
        return DEFAULT_PROB

    # ---------- Total goals (one half) ----------
    if qtype == "half_total_goals":
        if total_xg is not None:
            mu = total_xg * _half_share(parsed.get("half"))
            return prob_over_under(mu, parsed["threshold"], parsed["direction"])
        return DEFAULT_PROB

    # ---------- Second half vs first half goals ----------
    if qtype == "half_vs_half_goals":
        if total_xg is not None:
            mu_first = total_xg * FIRST_HALF_GOAL_SHARE
            mu_second = total_xg * SECOND_HALF_GOAL_SHARE
            if parsed.get("more_half") == "first":
                return prob_x_greater_than_y(mu_first, mu_second)
            return prob_x_greater_than_y(mu_second, mu_first)
        # No xG: second half historically outscores the first, but ties take
        # meaningful mass, so just below a coin flip for "strictly more".
        return 0.45

    # ---------- BTTS AND total goals over N ----------
    if qtype == "btts_and_total_goals":
        if xg_home is not None:
            # Exact bivariate-Poisson joint: both teams score AND total >= N.
            return prob_btts_and_over(xg_home, xg_away, parsed["threshold"])
        return 0.30

    # ---------- Halftime tied ----------
    if qtype == "halftime_tied":
        if xg_home is not None:
            _, tie, _ = halftime_outcome_probs(xg_home, xg_away)
            return tie
        return 0.40

    # ---------- Halftime winning ----------
    if qtype == "halftime_winning":
        if xg_home is not None:
            home_lead, _, away_lead = halftime_outcome_probs(xg_home, xg_away)
            is_home = team_is_home(match_name, parsed.get("team", ""))
            return home_lead if is_home else away_lead
        return 0.28

    # ---------- Halftime both teams have a shot on target ----------
    if qtype == "halftime_both_sot":
        if xg_home is not None:
            mu_h = (max(MIN_TEAM_SOT, xg_home * _sot_per_xg(xg_home))) * FIRST_HALF_GOAL_SHARE
            mu_a = (max(MIN_TEAM_SOT, xg_away * _sot_per_xg(xg_away))) * FIRST_HALF_GOAL_SHARE
            return prob_at_least_one(mu_h) * prob_at_least_one(mu_a)
        return 0.45

    # ---------- Player shot on target ----------
    if qtype == "player_shot_on_target":
        return _player_sot_prob(
            parsed.get("player", ""), parsed.get("threshold"),
            parsed.get("half"), match_name, xg_home, xg_away,
        )

    # ---------- Player goal involvement (goal or assist) ----------
    if qtype == "player_goal_involvement":
        player = parsed.get("player", "")
        context = _player_team_context(player, match_name, xg_home, xg_away)
        stats = _real_player_stats(player)
        if (stats and stats.get("shots_per_90", 0.0) >= MIN_REAL_STAT
                and stats.get("conversion_rate", 0.0) >= MIN_REAL_STAT):
            # Real stats: player's own rates, scaled only by team performance.
            player_xg = stats["shots_per_90"] * stats["conversion_rate"] * context
            p_goal = prob_at_least_one(player_xg)
            p_assist = prob_at_least_one(stats.get("xA_per_90", 0.0) * context)
        else:
            # No real stats: team-xG-derived estimate (already team-scaled).
            team_xg = _player_team_xg(player, match_name, xg_home, xg_away)
            if team_xg is None:
                return PLAYER_GOAL_INVOLVEMENT_BASE   # no information
            p_goal = prob_at_least_one(team_xg / TEAM_AVG_GOAL_SCORERS)
            p_assist = prob_at_least_one(team_xg * PLAYER_ASSIST_RATE)
        return 1 - (1 - p_goal) * (1 - p_assist)

    # ---------- Team total shots on target ----------
    if qtype == "team_total_sot":
        # Knockout player SOT markets ("<Player> have N or more shots on target")
        # that lacked a "(Country)" tag land here. If the subject isn't one of the
        # two teams, price it as a single player instead of a whole team.
        if match_name and _team_side(match_name, parsed.get("team", "")) is None:
            return _player_sot_prob(
                parsed.get("team", ""), parsed.get("threshold"),
                parsed.get("half"), match_name, xg_home, xg_away,
            )
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        eff_xg = team_xg if team_xg else 1.3
        mu = max(MIN_TEAM_SOT, eff_xg * _sot_per_xg(eff_xg))
        if parsed.get("half"):
            mu *= _half_share(parsed["half"])
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Total shots on target (match) ----------
    if qtype == "total_sot":
        if total_xg is not None:
            mu = max(2 * MIN_TEAM_SOT, total_xg * _sot_per_xg(total_xg / 2))
        else:
            mu = 9.0
        if parsed.get("half"):
            mu *= _half_share(parsed["half"])
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Team corners ----------
    if qtype == "team_corners":
        # Scale the base corner count by the team's attacking intent (xG):
        # attacking sides win more corners. Falls back to flat when no odds.
        base_mu = AVG_CORNERS_PER_TEAM
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        if team_xg is not None:
            xg_ratio = team_xg / AVG_TEAM_XG
            base_mu = AVG_CORNERS_PER_TEAM * (1 + CORNER_XG_SCALE * (xg_ratio - 1))
        mu = base_mu * (_half_share(parsed.get("half")) if parsed.get("half") else 1.0)
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Team offsides ----------
    if qtype == "team_offsides":
        # Scale the base offside rate by the team's attacking intent (xG):
        # attacking sides get caught offside more, defensive sides less. Falls
        # back to the flat base rate when no odds/xG are available.
        base_mu = AVG_OFFSIDES_PER_TEAM
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        if team_xg is not None:
            xg_ratio = team_xg / AVG_TEAM_XG
            base_mu = AVG_OFFSIDES_PER_TEAM * (1 + OFFSIDE_XG_SCALE * (xg_ratio - 1))
        mu = base_mu * (_half_share(parsed.get("half")) if parsed.get("half") else 1.0)
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Team cards ----------
    if qtype == "team_cards":
        mu = (AVG_CARDS_TOTAL / 2) * (_half_share(parsed.get("half")) if parsed.get("half") else 1.0)
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Total cards (match) ----------
    if qtype == "total_cards":
        mu = AVG_CARDS_TOTAL * (_half_share(parsed.get("half")) if parsed.get("half") else 1.0)
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Comparative: team more <metric> than opponent ----------
    if qtype == "team_more_than_opponent":
        metric = parsed["metric"]
        share = _half_share(parsed.get("half")) if parsed.get("half") else 1.0
        if metric == "shots on target" and xg_home is not None:
            is_home = team_is_home(match_name, parsed.get("team", ""))
            xg_t = xg_home if is_home else xg_away
            xg_o = xg_away if is_home else xg_home
            mu_team = max(MIN_TEAM_SOT, xg_t * _sot_per_xg(xg_t)) * share
            mu_opp = max(MIN_TEAM_SOT, xg_o * _sot_per_xg(xg_o)) * share
            return prob_x_greater_than_y(mu_team, mu_opp)
        if metric == "goals" and xg_home is not None:
            is_home = team_is_home(match_name, parsed.get("team", ""))
            mu_team = (xg_home if is_home else xg_away) * share
            mu_opp = (xg_away if is_home else xg_home) * share
            return prob_x_greater_than_y(mu_team, mu_opp)
        if metric == "corner kicks":
            # Slight edge to the favorite; otherwise near coin flip.
            return _comparative_with_supremacy(odds, match_name, parsed, AVG_CORNERS_PER_TEAM)
        if metric in ("fouls", "cards"):
            # Underdogs tend to commit marginally more fouls/cards.
            base = AVG_FOULS_PER_TEAM if metric == "fouls" else AVG_CARDS_TOTAL / 2
            return _comparative_with_supremacy(odds, match_name, parsed, base, favor_underdog=True)
        return 0.42  # P(strictly more) with a meaningful tie probability

    # ---------- Both teams to score (BTTS) ----------
    if qtype == "btts":
        if xg_home is not None:
            return predict_btts(xg_home, xg_away)
        return 0.50

    # ---------- Team scores N+ goals ----------
    if qtype == "team_goals_over":
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        if team_xg is not None:
            return prob_over_under(team_xg, parsed["threshold"], parsed.get("direction", "over"))
        # No xG: rough base rates by threshold.
        return {2: 0.42, 3: 0.18, 4: 0.07}.get(parsed["threshold"], 0.30)

    # ---------- Team scores in both halves ----------
    if qtype == "team_score_both_halves":
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        if team_xg is not None:
            p_first = prob_at_least_one(team_xg * FIRST_HALF_GOAL_SHARE)
            p_second = prob_at_least_one(team_xg * SECOND_HALF_GOAL_SHARE)
            return p_first * p_second
        return 0.30

    # ---------- Team keeps a clean sheet ----------
    if qtype == "team_clean_sheet":
        side = _team_side(match_name, parsed.get("team", ""))
        if side is not None and xg_home is not None:
            opp_xg = xg_away if side == "home" else xg_home
            return 1 - prob_at_least_one(opp_xg)   # P(opponent scores 0)
        return 0.30

    # ---------- Regulation ends in a draw ----------
    if qtype == "match_draw":
        return p_draw_sp if odds else 0.25

    # ---------- Advance to the next round ----------
    if qtype == "team_advance":
        if odds:
            side = _team_side(match_name, parsed.get("team", ""))
            p_win = p_home_sp if side == "home" else (p_away_sp if side == "away" else None)
            if p_win is not None:
                # Win in regulation, else ~coin flip after extra time / penalties.
                return min(0.99, p_win + p_draw_sp * 0.5)
        return DEFAULT_PROB

    # ---------- Team wins by N+ goals ----------
    if qtype == "team_win_by_margin":
        side = _team_side(match_name, parsed.get("team", ""))
        if side is not None and xg_home is not None:
            team_xg = xg_home if side == "home" else xg_away
            opp_xg = xg_away if side == "home" else xg_home
            return prob_win_by_margin(team_xg, opp_xg, parsed.get("margin", 2))
        return 0.20

    # ---------- Total shots (on + off target) ----------
    if qtype == "total_shots":
        if total_xg is not None:
            mu = TOTAL_SHOTS_BASELINE * (total_xg / (2 * AVG_TEAM_XG))
        else:
            mu = TOTAL_SHOTS_BASELINE
        if parsed.get("half"):
            mu *= _half_share(parsed["half"])
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Total corners (both teams) ----------
    if qtype == "total_corners":
        # Scale the base corner count by the match's combined attacking intent
        # (total xG vs an average match). Falls back to flat when no odds.
        base_mu = AVG_TOTAL_CORNERS
        if total_xg is not None:
            xg_ratio = total_xg / (2 * AVG_TEAM_XG)
            base_mu = AVG_TOTAL_CORNERS * (1 + CORNER_XG_SCALE * (xg_ratio - 1))
        mu = base_mu * (_half_share(parsed.get("half")) if parsed.get("half") else 1.0)
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Total offsides (both teams) ----------
    if qtype == "total_offsides":
        mu = 2 * AVG_OFFSIDES_PER_TEAM * (_half_share(parsed.get("half")) if parsed.get("half") else 1.0)
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Both teams receive a card ----------
    if qtype == "both_teams_card":
        p_one = prob_at_least_one(AVG_CARDS_TOTAL / 2)
        return p_one * p_one

    # ---------- A card shown in the first half ----------
    if qtype == "card_first_half":
        return prob_at_least_one(AVG_CARDS_TOTAL * CARD_FIRST_HALF_SHARE)

    # ---------- A card shown late (after the 2nd hydration break / incl. ET) ----------
    if qtype == "card_late":
        return CARD_LATE_RATE

    # ---------- A red card shown ----------
    if qtype == "red_card":
        return RED_CARD_RATE

    # ---------- A substitute scores ----------
    if qtype == "sub_scores":
        return SUB_SCORES_RATE

    # ---------- A substitution before halftime ----------
    if qtype == "sub_before_half":
        return SUB_BEFORE_HALF_RATE

    # ---------- Any player records 2+ shots on target ----------
    if qtype == "any_player_sot":
        return ANY_PLAYER_MULTI_SOT_RATE

    # ---------- Any player scores 2+ goals (a brace) ----------
    if qtype == "any_player_brace":
        return ANY_PLAYER_BRACE_RATE

    # ---------- Goal scored inside a specific time window ----------
    if qtype in ("goal_before_hydration", "goal_after_hydration",
                 "goal_first_half_stoppage", "goal_second_half_stoppage"):
        frac = {
            "goal_before_hydration": GOAL_FRAC_BEFORE_HYDRATION,
            "goal_after_hydration": GOAL_FRAC_AFTER_HYDRATION,
            "goal_first_half_stoppage": GOAL_FRAC_FIRST_HALF_STOPPAGE,
            "goal_second_half_stoppage": GOAL_FRAC_SECOND_HALF_STOPPAGE,
        }[qtype]
        if total_xg is not None:
            return prob_at_least_one(total_xg * frac)
        return GOAL_WINDOW_FALLBACK[qtype.replace("goal_", "")]

    # ---------- Penalty markets ----------
    if qtype == "penalty_awarded":
        return PENALTY_AWARDED_RATE
    if qtype == "penalty_or_red_card":
        return PENALTY_OR_RED_RATE

    # ---------- Unknown ----------
    return DEFAULT_PROB


def _comparative_with_supremacy(odds, match_name, parsed, base_mu, favor_underdog=False):
    """
    P(team has strictly more of a count metric than opponent), starting from a
    coin-flip-minus-ties baseline and nudging by match supremacy.
    """
    from model.poisson import prob_x_greater_than_y
    # Equal means -> P(more) for two equal Poissons.
    base = prob_x_greater_than_y(base_mu, base_mu)
    if not odds:
        return base
    # Align odds to the SportsPredict ordering before computing supremacy.
    p_home_sp, _, p_away_sp, _ = _aligned_odds(odds, match_name)
    is_home = team_is_home(match_name, parsed.get("team", ""))
    supremacy = (p_home_sp - p_away_sp) if is_home else (p_away_sp - p_home_sp)
    if favor_underdog:
        supremacy = -supremacy
    # Nudge by up to ~0.12 based on supremacy.
    return max(0.05, min(0.95, base + supremacy * 0.12))


def run_model_on_market(market, odds_index):
    prob = _model_prob_for_market(market, odds_index)
    # Shrink signal-less peripheral markets toward 50 to limit Brier downside.
    qtype = parse_question(market.get("question", "")).get("type")
    if qtype in PERIPHERAL_TYPES:
        keep = PERIPHERAL_SHRINK_OVERRIDES.get(qtype, PERIPHERAL_SHRINK)
        prob = _shrink_to_half(prob, keep)
    return format_prediction_for_submission(prob)


def adjust_for_lineup(market, current_int_prob, starters):
    """
    Given confirmed `starters` (a set of normalized starting-XI names), adjust a
    player market's already-submitted integer probability (1-99):

      - player markets only (non-player markets returned unchanged),
      - if the named player is NOT in the starting XI -> slash to
        max(1, round(prob * BENCH_PLAYER_FACTOR)),
      - if the player name can't be matched at all -> leave unchanged.

    Returns the new integer probability (1-99).
    """
    from data.fetch_lineups import player_is_starter

    parsed = parse_question(market.get("question", ""))
    if parsed.get("type") not in ("player_shot_on_target", "player_goal_involvement"):
        return current_int_prob
    if not starters:
        return current_int_prob

    player = parsed.get("player", "")
    is_starter = player_is_starter(player, starters)
    if is_starter is None:
        return current_int_prob          # no data / unmatched -> unchanged
    if is_starter:
        return current_int_prob          # starting -> keep our prior
    # Benched or absent: slash hard.
    return max(1, round(current_int_prob * BENCH_PLAYER_FACTOR))


def run_submission_loop():
    log = logging.getLogger(__name__)
    try:
        log.info("=== Submission run started ===")
        reset_player_budget()
        event, lobby = get_probability_cup_lobby_and_event()
        lobby_id = lobby["id"]
        event_id = event["id"]

        join_lobby(lobby_id)
        log.info("Joined lobby %s", lobby_id)

        matches = fetch_matches(event_id, lobby_id)
        log.info("Got %d matches, fetching markets...", len(matches))
        markets = []
        for match in matches:
            mks, _ = fetch_markets(lobby_id, match["id"])
            markets.extend(mks)
        log.info("Found %d total markets", len(markets))

        odds_index = build_odds_index(fetch_market_odds())
        log.info("Indexed odds for %d matches", len(odds_index))

        existing = {p["market_id"]: p for p in fetch_my_predictions(lobby_id)}
        log.info("Existing predictions on record: %d", len(existing))

        new_predictions = []
        patch_count = 0
        patch_failed = 0

        for m in markets:
            prob = run_model_on_market(m, odds_index)
            market_id = m["id"]

            if market_id in existing:
                pred_id = existing[market_id]["id"]
                try:
                    update_prediction(pred_id, prob)
                    patch_count += 1
                    if patch_count % 10 == 0:
                        log.info("  Patched %d/%d...", patch_count, len(existing))
                except Exception as exc:
                    log.warning("[patch] Failed %s: %s", market_id, exc)
                    patch_failed += 1
            else:
                new_predictions.append({
                    "market_id": market_id,
                    "lobby_id": lobby_id,
                    "probability": prob,
                })

        for i in range(0, len(new_predictions), 50):
            chunk = new_predictions[i:i + 50]
            submit_predictions_batch(chunk)
            log.info("Submitted batch of %d new predictions", len(chunk))

        log.info("Updated %d predictions via PATCH (%d failed)", patch_count, patch_failed)
        log.info("=== Submission run completed successfully ===")
    except Exception as e:
        logging.getLogger(__name__).error("Error during submission loop: %s", e, exc_info=True)


# --- Lineup-triggered updates -----------------------------------------------
# Matches whose confirmed lineup has already been consumed (so we stop polling
# them and don't waste api-football quota). Keyed by SportsPredict match id.
_lineup_done = set()


def _parse_iso(ts):
    """Parse an ISO-8601 timestamp (with trailing 'Z') into an aware datetime."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _match_in_lineup_window(match, now):
    """
    True if `now` is within [kickoff - LINEUP_WINDOW_MINUTES, kickoff) for this
    match. Kickoff is the match's closing_time (markets lock at kickoff).
    """
    kickoff = _parse_iso(match.get("closing_time") or match.get("opening_time"))
    if kickoff is None:
        return False, None
    window_start = kickoff - timedelta(minutes=LINEUP_WINDOW_MINUTES)
    return (window_start <= now < kickoff), kickoff


def run_lineup_updates(now=None):
    """
    One lineup-check sweep. For every match inside its pre-kickoff window whose
    lineup we haven't consumed yet:
      - resolve the api-football fixture id,
      - fetch the confirmed starting XI,
      - if available, re-run the model on that match's open markets and PATCH,
        slashing player markets for benched/absent players,
      - mark the match done so we stop polling it.

    On HTTP 429 from api-football, the whole sweep is skipped (retried next
    cycle). Safe to call repeatedly from a scheduler.
    """
    from data.fetch_lineups import fetch_lineup, resolve_fixture_id, LineupRateLimited

    log = logging.getLogger(__name__)
    now = now or datetime.now(timezone.utc)

    try:
        event, lobby = get_probability_cup_lobby_and_event()
        lobby_id = lobby["id"]
        join_lobby(lobby_id)
        matches = fetch_matches(event["id"], lobby_id)
    except Exception as exc:
        log.error("Lineup sweep: could not load matches: %s", exc)
        return

    # Only matches currently in their pre-kickoff window and not yet consumed.
    pending = []
    for m in matches:
        if m["id"] in _lineup_done:
            continue
        in_window, kickoff = _match_in_lineup_window(m, now)
        if kickoff is None:
            log.warning("Lineup check: missing kickoff_time for %s", m.get("name"))
            continue
        if in_window:
            pending.append(m)

    if not pending:
        log.info("Lineup sweep: no matches in the pre-kickoff window.")
        return

    odds_index = build_odds_index(fetch_market_odds())

    for m in pending:
        match_name = m.get("name", "")
        fixture_id = None
        try:
            fixture_id = resolve_fixture_id(match_name)
        except LineupRateLimited:
            log.warning("Lineup sweep: rate-limited resolving fixtures; skipping cycle.")
            return
        if fixture_id is None:
            log.info("Lineup check: could not resolve fixture for %s", match_name)
            continue

        try:
            starters = fetch_lineup(fixture_id)
        except LineupRateLimited:
            log.warning("Lineup sweep: 429 fetching lineup; skipping rest of cycle.")
            return

        if not starters:
            log.info("Lineup check: no lineup available for %s", match_name)
            continue

        # Lineup is in — re-run the model and PATCH this match's open markets.
        try:
            markets, _ = fetch_markets(lobby_id, m["id"])
            existing = {p["market_id"]: p for p in fetch_my_predictions(lobby_id)}
        except Exception as exc:
            log.warning("Lineup update: could not load markets for %s: %s", match_name, exc)
            continue

        patched = 0
        for market in markets:
            base_prob = run_model_on_market(market, odds_index)
            final_prob = adjust_for_lineup(market, base_prob, starters)
            pred = existing.get(market["id"])
            if not pred:
                continue
            try:
                update_prediction(pred["id"], final_prob)
                patched += 1
            except Exception as exc:
                log.warning("Lineup update PATCH failed for %s: %s", market["id"], exc)

        _lineup_done.add(m["id"])
        log.info("Lineup update: %s — patched %d markets from confirmed XI.",
                 match_name, patched)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_submission_loop()
