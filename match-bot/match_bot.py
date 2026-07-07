"""
Match-bot — thin wrapper around the competition-bot pipeline.

It reuses competition-bot's model, data fetching, and SportsPredict client
unchanged. It adds exactly two things on top:

  1. A second SportsPredict API key (SPORTSPREDICT_KEY_BOT2) so this runs as a
     separate bot on the same account.
  2. A post-processing step that extremizes the model's probabilities on the
     markets with genuine signal, and removes peripheral shrinkage.

Run it with ``python scheduler.py`` (loops on a timer) or ``python match_bot.py``
(single submission pass).
"""

import os
import sys
import logging

# --- Make the competition-bot package importable ----------------------------
# The match-bot reuses competition-bot's model/data/client code wholesale, so
# we put that package directory on the import path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMPETITION_BOT = os.path.abspath(os.path.join(_HERE, "..", "competition-bot"))
if _COMPETITION_BOT not in sys.path:
    sys.path.insert(0, _COMPETITION_BOT)

from dotenv import load_dotenv

# competition-bot modules (resolved via the path insert above)
import bot.client as client
from bot.client import SportsPredictClient
from bot.question_parser import parse_question
from bot.submit import (
    _model_prob_for_market,
    _match_name,
    reset_player_budget,
    adjust_for_lineup,
    LINEUP_CHECK_INTERVAL_MINUTES,
)
from model.ensemble import format_prediction_for_submission
from data.fetch_odds import fetch_market_odds
from bot.match_data import build_odds_index

from config import (
    BINARY_TYPES,
    HEAVY_EXTREMIZE_K,
    HEAVY_EXTREMIZE_TYPES,
    LIGHT_EXTREMIZE_K,
    TARGET_MATCH,
    API_KEY_ENV,
)
from extremize import extremize

load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))
load_dotenv()  # also honour a local .env if present

logger = logging.getLogger(__name__)


def configure_client():
    """
    Point the shared SportsPredict client at the match-bot's own API key.

    The competition allows two bots per account, each with its own key. Reusing
    the competition-bot's key here would make both bots submit as one entry, so
    we require a distinct key and fail loudly if it's missing.
    """
    key = os.getenv(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. The match-bot needs its own SportsPredict "
            f"API key (distinct from the competition-bot's SPORTSPREDICT_KEY). "
            f"Add {API_KEY_ENV}=<your second key> to the project .env."
        )
    if key == os.getenv("SPORTSPREDICT_KEY"):
        raise RuntimeError(
            f"{API_KEY_ENV} is identical to SPORTSPREDICT_KEY. The match-bot must "
            f"use a different key so it runs as a separate bot entry."
        )
    client._default_client = SportsPredictClient(api_key=key)
    logger.info("Match-bot client configured with %s", API_KEY_ENV)


def _target_match_matches(match_name):
    """True if this market's match should be submitted given TARGET_MATCH."""
    if TARGET_MATCH is None:
        return True
    return TARGET_MATCH.strip().lower() in (match_name or "").strip().lower()


def run_model_on_market(market, odds_index):
    """
    Match-bot three-tier prediction strategy:

      Tier 1 (BINARY_TYPES)         → submit 1 or 99
      Tier 2 (HEAVY_EXTREMIZE_TYPES)→ logit stretch k=2.5
      Tier 3 (everything else)      → light logit stretch k=1.5

    Tier 1 contains only types where calibration shows the model has been
    directionally right and produced positive RBP vs crowd. Tier 3 covers
    player markets and other types that have been unreliable.
    """
    qtype = parse_question(market.get("question", "")).get("type")
    prob = _model_prob_for_market(market, odds_index)

    if qtype in BINARY_TYPES:
        return 99 if prob > 0.50 else 1 if prob < 0.50 else 50
    elif qtype in HEAVY_EXTREMIZE_TYPES:
        return format_prediction_for_submission(extremize(prob, HEAVY_EXTREMIZE_K))
    else:
        return format_prediction_for_submission(extremize(prob, LIGHT_EXTREMIZE_K))


def run_submission_loop():
    """
    One full submission pass for the match-bot.

    Mirrors competition-bot's submission loop but (a) uses the extremizing model
    function above and (b) honours TARGET_MATCH so the bot can focus on a single
    high-conviction match.
    """
    from bot.client import (
        get_probability_cup_lobby_and_event,
        fetch_matches,
        fetch_markets,
        fetch_my_predictions,
        submit_predictions_batch,
        update_prediction,
        join_lobby,
    )

    log = logging.getLogger(__name__)
    try:
        log.info("=== Match-bot submission run started ===")
        reset_player_budget()
        event, lobby = get_probability_cup_lobby_and_event()
        lobby_id = lobby["id"]
        event_id = event["id"]

        join_lobby(lobby_id)
        log.info("Joined lobby %s", lobby_id)

        matches = fetch_matches(event_id, lobby_id)
        if TARGET_MATCH is not None:
            matches = [m for m in matches
                       if _target_match_matches(m.get("name", ""))]
            log.info("TARGET_MATCH=%r -> %d match(es) selected",
                     TARGET_MATCH, len(matches))
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

        log.info("Updated %d predictions via PATCH (%d failed)",
                 patch_count, patch_failed)
        log.info("=== Match-bot submission run completed successfully ===")
    except Exception as e:
        log.error("Error during match-bot submission loop: %s", e, exc_info=True)


def run_lineup_updates(now=None):
    """
    Lineup-poll sweep for the match-bot.

    Reuses competition-bot's lineup logic but swaps in the extremizing model
    function and the TARGET_MATCH filter by temporarily patching the
    competition-bot submit module's run_model_on_market.
    """
    import bot.submit as cb_submit

    original = cb_submit.run_model_on_market
    cb_submit.run_model_on_market = run_model_on_market
    try:
        cb_submit.run_lineup_updates(now=now)
    finally:
        cb_submit.run_model_on_market = original


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    configure_client()
    run_submission_loop()
