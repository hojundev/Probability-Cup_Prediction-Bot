import logging

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
)
from data.fetch_odds import fetch_market_odds
from data.fetch_player_stats import fetch_player_stats
from model.poisson import (
    predict_match_outcome,
    predict_btts,
    prob_at_least_one,
    prob_over_under,
    prob_total_goals,
    halftime_outcome_probs,
    prob_x_greater_than_y,
)
from model.ensemble import blend_probabilities, format_prediction_for_submission

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
PENALTY_OR_RED_RATE = 0.40     # P(penalty OR red card)

TEAM_SHOTS_PER_GAME = 13.0
TEAM_SHOTS_ON_TARGET_PER_GAME = 4.5

# Weight on the (sharp) betting-market line when blending with the model.
# Sharp books are sharper than the forecasting crowd, so on odds-anchored
# markets we ride the market hard and add only a little model.
MARKET_ALPHA = 0.88
DEFAULT_PROB = 0.50

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


def _model_prob_for_market(market, odds_index):
    """Route a market to the right model and return a decimal probability (0-1)."""
    question = market.get("question", "")
    parsed = parse_question(question)
    qtype = parsed["type"]
    match_name = _match_name(market)

    odds = find_match_odds(odds_index, match_name) if match_name else None
    xg_home = xg_away = None
    total_xg = None
    if odds:
        xg_home, xg_away = estimate_team_xg(
            odds["p_home"], odds["p_draw"], odds["p_away"], odds.get("total_goals")
        )
        total_xg = xg_home + xg_away

    # ---------- Match winner ----------
    if qtype == "match_winner":
        team = parsed.get("team", "")
        if odds:
            is_home = team_is_home(match_name, team)
            market_prob = odds["p_home"] if is_home else odds["p_away"]
            outcome = predict_match_outcome(xg_home, xg_away)
            model_prob = outcome["home_win"] if is_home else outcome["away_win"]
            return blend_probabilities(market_prob, model_prob, alpha=MARKET_ALPHA)
        return DEFAULT_PROB

    # ---------- Team scores (full match) ----------
    if qtype == "team_score":
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        return prob_at_least_one(team_xg) if team_xg else 0.62  # base rate a team scores

    # ---------- Team scores in a half ----------
    if qtype == "team_score_half":
        team_xg = _team_xg_for(parsed, match_name, xg_home, xg_away)
        if team_xg is None:
            return 0.40
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
            return team_xg / total_xg if total_xg > 0 else DEFAULT_PROB
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

    # ---------- BTTS AND total goals over N ----------
    if qtype == "btts_and_total_goals":
        if xg_home is not None:
            p_btts = predict_btts(xg_home, xg_away)
            p_total = prob_total_goals(xg_home, xg_away, parsed["threshold"], parsed["direction"])
            # Positively correlated; approximate joint as the smaller leg
            # nudged toward their product.
            return min(p_btts, p_total) * 0.85
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
        stats = fetch_player_stats(parsed.get("player", ""))
        # The player's own per-90 SOT rate is the expected count; scale by the
        # half share when the question is half-specific.
        player_xsot = stats["shots_on_target_per_90"]
        if parsed.get("half"):
            player_xsot *= _half_share(parsed["half"])
        return prob_at_least_one(player_xsot)

    # ---------- Player goal involvement (goal or assist) ----------
    if qtype == "player_goal_involvement":
        stats = fetch_player_stats(parsed.get("player", ""))
        player_xg = stats["shots_per_90"] * stats["conversion_rate"]
        p_goal = prob_at_least_one(player_xg)
        p_assist = prob_at_least_one(stats["xA_per_90"])
        return 1 - (1 - p_goal) * (1 - p_assist)

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
        return prob_over_under(AVG_CORNERS_PER_TEAM, parsed["threshold"], parsed["direction"])

    # ---------- Team offsides ----------
    if qtype == "team_offsides":
        return prob_over_under(AVG_OFFSIDES_PER_TEAM, parsed["threshold"], parsed["direction"])

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
    is_home = team_is_home(match_name, parsed.get("team", ""))
    supremacy = (odds["p_home"] - odds["p_away"]) if is_home else (odds["p_away"] - odds["p_home"])
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


def run_submission_loop():
    log = logging.getLogger(__name__)
    try:
        log.info("=== Submission run started ===")
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_submission_loop()
