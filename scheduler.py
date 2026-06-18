import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from bot.submit import run_submission_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

scheduler = BlockingScheduler()

# Run the bot every 2 hours.
# Each run: fetches open markets, runs the model, PATCHes/POSTs predictions.
# Odds are cached to disk for 2 hours so The Odds API is only called once per
# scheduler cycle (not once per SportsPredict API call).
@scheduler.scheduled_job('interval', hours=2)
def run_bot():
    logging.info("Scheduled job started: Running bot submission loop...")
    run_submission_loop()
    logging.info("Scheduled job finished.")

if __name__ == "__main__":
    logging.info("Starting scheduler. Press Ctrl+C to exit.")
    
    # Run once immediately on startup
    run_bot()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")
