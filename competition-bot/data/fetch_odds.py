"""
Fetches live betting market odds from The Odds API for World Cup matches.

To conserve the free-tier quota (500 requests/month) the response is cached
to disk for 2 hours. On each call we check the cache age first:
  - Cache younger than 2 hours  -> return cached data, no API call
  - Cache older than 2 hours    -> call the API, refresh the cache

This reduces usage from ~48 calls/day (every 30 min) to ~12 calls/day, well
within the 500/month budget for the full tournament.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"

# Cache file lives next to this module.
_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".odds_cache.json")

# How long to treat cached odds as fresh (seconds).
CACHE_TTL_SECONDS = 2 * 60 * 60  # 2 hours


def _load_cache():
    """Return (timestamp, data) from disk, or (0, None) if missing/corrupt."""
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("timestamp", 0), payload.get("data")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0, None


def _save_cache(data):
    """Persist odds data to disk with the current timestamp."""
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump({"timestamp": time.time(), "data": data}, fh)
    except OSError as exc:
        print(f"[fetch_odds] Could not write cache: {exc}")


def _fetch_from_api():
    """Call The Odds API and return the raw list, or [] on failure."""
    if not ODDS_API_KEY:
        print("[fetch_odds] Warning: ODDS_API_KEY not set. Returning empty odds.")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us,uk,eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    response = requests.get(ODDS_API_URL, params=params, timeout=15)
    if response.status_code == 200:
        data = response.json()
        # Log remaining quota so we can monitor usage.
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        if remaining is not None:
            print(f"[fetch_odds] API call succeeded. Quota: {used} used, {remaining} remaining.")
        return data
    else:
        print(f"[fetch_odds] Failed to fetch odds: {response.status_code} {response.text[:200]}")
        return []


def fetch_market_odds(force_refresh=False):
    """
    Return betting market odds for World Cup matches.

    Uses the disk cache if it is younger than 2 hours.
    Pass force_refresh=True to bypass the cache (e.g. just before a match).
    """
    if not force_refresh:
        timestamp, cached_data = _load_cache()
        age = time.time() - timestamp
        if cached_data is not None and age < CACHE_TTL_SECONDS:
            age_min = int(age // 60)
            print(f"[fetch_odds] Using cached odds ({age_min}m old, TTL=120m).")
            return cached_data

    # Cache is stale or missing — fetch fresh data.
    print("[fetch_odds] Fetching fresh odds from API...")
    data = _fetch_from_api()
    if data:
        _save_cache(data)
    elif True:
        # API call failed — fall back to stale cache rather than returning nothing.
        _, cached_data = _load_cache()
        if cached_data:
            print("[fetch_odds] API call failed; using stale cache as fallback.")
            return cached_data

    return data


if __name__ == "__main__":
    odds = fetch_market_odds()
    print(f"Fetched odds for {len(odds)} matches.")
    if odds:
        first = odds[0]
        print(f"Sample: {first.get('home_team')} vs {first.get('away_team')}")
