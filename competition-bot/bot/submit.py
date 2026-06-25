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
from data.fetch_player_stats import fetch_player_stats
from model.poisson import (
    predict_btts,
    prob_at_least_one,
    prob_over_under,
    prob_total_goals,
    prob_btts_and_over,
    halftime_outcome_probs,
    prob_x_greater_than_y,
)
from model.ensemble import blend_probabilities, format_prediction_for_submission
from model.elo import load_wc2026_ratings

logger = logging.getLogger(__name__)

# Share of goals scored in each half (World Cup historical split).
FIRST_HALF_GOAL_SHARE = 0.42
SECOND_HALF_GOAL_SHARE = 0.58

# Convert a team's expected goals into expected shots on target. Empirically a
# team puts roughly 3 shots on target per expected goal, with a floor.
SOT_PER_XG = 3.0
MIN_TEAM_SOT = 2.5

# Historical per-match base rates used as calibrated priors for peripheral
# markets the bookmaker feed doesn't cover. These beat a naive 0.50.
AVG_CORNERS_PER_TEAM = 5.0
AVG_OFFSIDES_PER_TEAM = 1.7
AVG_CARDS_TOTAL = 4.2          # yellow + red, full match
AVG_FOULS_PER_TEAM = 12.0
PENALTY_AWARDED_RATE = 0.26    # P(>=1 penalty in a match)
PENALTY_OR_RED_RATE = 0.37     # P(penalty OR red card)

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

# --- Player market baseline blending ----------------------------------------
# WC player markets resolve 0 (player didn't score/assist/get SOT) the vast
# majority of the time, even for good players. The model's per-90 stats and
# xG-based fallbacks tend to over-predict because they assume club-level
# scoring rates in a tight tournament match. We blend the model output with a
# conservative WC baseline to anchor predictions downward and beat the crowd
# (which tends to cluster around 40%).
#
# final = PLAYER_BASELINE_WEIGHT * baseline + (1 - PLAYER_BASELINE_WEIGHT) * model_prob
#
# Tuning guide:
#   - Increase PLAYER_BASELINE_WEIGHT to lean harder on the baseline (more conservative)
#   - Decrease it to trust the model more (more differentiation between players)
PLAYER_GOAL_INVOLVEMENT_BASELINE = 0.15   # conservative WC base rate for goal or assist
PLAYER_SOT_BASELINE = 0.25                # conservative WC base rate for shot on target
PLAYER_BASELINE_WEIGHT = 0.60             # fraction of final prob pulled from baseline

# Player markets for absent/benched players are slashed to this fraction of the
# starter probability once confirmed lineups are in (see lineup updates).
BENCH_PLAYER_FACTOR = 0.30

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
    "team_more_than_opponent",
}


def _shrink_to_half(prob, keep):
    """Pull a probability toward 0.50, keeping `keep` of its deviation."""
    return 0.50 + (prob - 0.50) * keep


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
      - the per-run lookup budget is exhausted,
      - api-football has no real data (is_real is False), or
      - the stats are present but effectively zero (below MIN_REAL_STAT).
    """
    if not player_name or player_name == "Unknown":
        return None
    key = player_name.strip().lower()
    if key not in _player_request_names:
        if len(_player_request_names) >= MAX_PLAYER_REQUESTS_PER_RUN:
            return None
        _player_request_names.add(key)
    stats = fetch_player_stats(player_name)
    if not stats or not stats.get("is_real"):
        return None
    return stats


def _model_prob_for_market(market, odds_index):
    """Route a market to the right model and return a decimal probability (0-1)."""
    question = market.get("question", "")
    parsed = parse_question(question)
    qtype = parsed["type"]
    match_name = _match_name(market)

    odds = find_match_odds(odds_index, match_name) if match_name else None
    xg_home = xg_away = None
    total_xg = None
    p_home_sp = p_away_sp = None
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
            mu_h = (max(MIN_TEAM_SOT, xg_home * SOT_PER_XG)) * FIRST_HALF_GOAL_SHARE
            mu_a = (max(MIN_TEAM_SOT, xg_away * SOT_PER_XG)) * FIRST_HALF_GOAL_SHARE
            return prob_at_least_one(mu_h) * prob_at_least_one(mu_a)
        return 0.45

    # ---------- Player shot on target ----------
    if qtype == "player_shot_on_target":
        player = parsed.get("player", "")
        stats = _real_player_stats(player)
        if stats and stats.get("shots_on_target_per_90", 0.0) >= MIN_REAL_STAT:
            player_xsot = stats["shots_on_target_per_90"]
        else:
            # No real stats: try to price off the player's actual team xG.
            team_xg = _player_team_xg(player, match_name, xg_home, xg_away)
            if team_xg is None:
                # Team unidentifiable -> conservative base rate (shrunk to 50),
                # scaled down for half-specific questions.
                base = PLAYER_SOT_BASE
                if parsed.get("half"):
                    base *= _half_share(parsed["half"])
                return _shrink_to_half(base, 0.6)
            player_xsot = (team_xg * SOT_PER_XG) / TEAM_AVG_PLAYERS_SHOOTING
        if parsed.get("half"):
            player_xsot *= _half_share(parsed["half"])
        model_prob = prob_at_least_one(player_xsot)
        # Blend with conservative WC baseline — model alone over-predicts because
        # club per-90 stats don't translate directly to tight tournament matches.
        return PLAYER_BASELINE_WEIGHT * PLAYER_SOT_BASELINE + (1 - PLAYER_BASELINE_WEIGHT) * model_prob

    # ---------- Player goal involvement (goal or assist) ----------
    if qtype == "player_goal_involvement":
        player = parsed.get("player", "")
        stats = _real_player_stats(player)
        if (stats and stats.get("shots_per_90", 0.0) >= MIN_REAL_STAT
                and stats.get("conversion_rate", 0.0) >= MIN_REAL_STAT):
            player_xg = stats["shots_per_90"] * stats["conversion_rate"]
            p_goal = prob_at_least_one(player_xg)
            p_assist = prob_at_least_one(stats.get("xA_per_90", 0.0))
        else:
            # No real stats: price off the player's actual team xG if known.
            team_xg = _player_team_xg(player, match_name, xg_home, xg_away)
            if team_xg is None:
                # Team unidentifiable -> conservative base rate (shrunk to 50).
                return _shrink_to_half(PLAYER_GOAL_INVOLVEMENT_BASE, 0.6)
            p_goal = prob_at_least_one(team_xg / TEAM_AVG_GOAL_SCORERS)
            p_assist = prob_at_least_one(team_xg * PLAYER_ASSIST_RATE)
        model_prob = 1 - (1 - p_goal) * (1 - p_assist)
        # Blend with conservative WC baseline — model alone over-predicts because
        # club per-90 stats don't translate directly to tight tournament matches.
        return PLAYER_BASELINE_WEIGHT * PLAYER_GOAL_INVOLVEMENT_BASELINE + (1 - PLAYER_BASELINE_WEIGHT) * model_prob

    # ---------- Team total shots on target ----------
    if qtype == "team_total_sot":
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        mu = max(MIN_TEAM_SOT, (team_xg if team_xg else 1.3) * SOT_PER_XG)
        if parsed.get("half"):
            mu *= _half_share(parsed["half"])
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Total shots on target (match) ----------
    if qtype == "total_sot":
        if total_xg is not None:
            mu = max(2 * MIN_TEAM_SOT, total_xg * SOT_PER_XG)
        else:
            mu = 9.0
        if parsed.get("half"):
            mu *= _half_share(parsed["half"])
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Team corners ----------
    if qtype == "team_corners":
        mu = AVG_CORNERS_PER_TEAM * (_half_share(parsed.get("half")) if parsed.get("half") else 1.0)
        return prob_over_under(mu, parsed["threshold"], parsed["direction"])

    # ---------- Team offsides ----------
    if qtype == "team_offsides":
        mu = AVG_OFFSIDES_PER_TEAM * (_half_share(parsed.get("half")) if parsed.get("half") else 1.0)
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
            mu_team = max(MIN_TEAM_SOT, (xg_home if is_home else xg_away) * SOT_PER_XG) * share
            mu_opp = max(MIN_TEAM_SOT, (xg_away if is_home else xg_home) * SOT_PER_XG) * share
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
        prob = _shrink_to_half(prob, PERIPHERAL_SHRINK)
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
