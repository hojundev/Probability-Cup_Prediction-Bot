"""
Match-bot scheduler.

A full model sweep every 2 hours, submitted under the match-bot's own API key
with extremized probabilities.

NOTE: unlike the competition-bot, this scheduler intentionally does NOT run a
lineup poll. Lineup polling hits api-football (100 req/day, shared with the
competition-bot) and is not disk-cached, so a second poller would roughly
double that usage. The match-bot also deliberately does not extremize player
markets, so confirmed-lineup slashing adds little value here. If you ever want
it, call match_bot.run_lineup_updates() manually.
"""

import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from match_bot import configure_client, run_submission_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

scheduler = BlockingScheduler()


@scheduler.scheduled_job("interval", hours=2)
def run_bot():
    logging.info("Scheduled job started: Running match-bot submission loop...")
    run_submission_loop()
    logging.info("Scheduled job finished.")


if __name__ == "__main__":
    logging.info("Starting match-bot scheduler. Press Ctrl+C to exit.")

    # Configure the second API key before any request goes out.
    configure_client()

    # Run a full submission once immediately on startup.
    run_bot()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Match-bot scheduler stopped.")
