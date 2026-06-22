import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from bot.submit import run_submission_loop, run_lineup_updates, LINEUP_CHECK_INTERVAL_MINUTES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

scheduler = BlockingScheduler()


# Full model sweep every 2 hours.
# Each run: fetches open markets, runs the model, PATCHes/POSTs predictions.
# Odds are cached to disk for 2 hours so The Odds API is only called once per
# scheduler cycle (not once per SportsPredict API call).
@scheduler.scheduled_job('interval', hours=2)
def run_bot():
    logging.info("Scheduled job started: Running bot submission loop...")
    run_submission_loop()
    logging.info("Scheduled job finished.")


# Lineup-poll sweep every 15 minutes.
# Checks each match inside its 90-minute pre-kickoff window for a confirmed
# starting XI. When lineups drop, it re-runs the model for that match and
# slashes player markets for anyone benched/absent, then stops polling it.
# Cheap when no lineups are out (one fixture lookup is cached; matches outside
# the window are skipped without any api-football call).
@scheduler.scheduled_job('interval', minutes=LINEUP_CHECK_INTERVAL_MINUTES)
def poll_lineups():
    logging.info("Lineup poll started...")
    run_lineup_updates()
    logging.info("Lineup poll finished.")


if __name__ == "__main__":
    logging.info("Starting scheduler. Press Ctrl+C to exit.")

    # Run a full submission once immediately on startup.
    run_bot()
    # Then do an immediate lineup sweep so we don't wait 15 min for the first.
    poll_lineups()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")
