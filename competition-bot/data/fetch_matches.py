"""
Fetches upcoming World Cup matches from football-data.org (v4).

Rate limiting (free tier):
  The API returns two headers on every response that we use for automatic
  throttling:
    X-Requests-Available-Minute  — requests remaining in the current 60-s window
    X-RequestCounter-Reset       — seconds until the window resets

  _throttle() reads these after every call. If we're down to 1 request left it
  sleeps until the window resets before returning, so the next call is safe.
  A 429 response is also handled with an automatic back-off and retry.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")
FOOTBALL_DATA_API_URL = "https://api.football-data.org/v4"

# Leave this many requests in reserve before sleeping.
# Set to 1 so we sleep only when we've used all but the very last slot.
RATE_LIMIT_RESERVE = 1

# Extra buffer seconds added on top of the reset window when sleeping.
SLEEP_BUFFER_SECONDS = 2


def _throttle(response: requests.Response) -> None:
    """
    Inspect rate-limit headers from a successful response and sleep if we're
    about to exhaust the per-minute quota.
    """
    try:
        available = int(response.headers.get("X-Requests-Available-Minute", 99))
        reset_in = int(response.headers.get("X-RequestCounter-Reset", 60))
    except (TypeError, ValueError):
        return

    if available <= RATE_LIMIT_RESERVE:
        sleep_for = reset_in + SLEEP_BUFFER_SECONDS
        print(
            f"[fetch_matches] Rate limit low ({available} requests left). "
            f"Sleeping {sleep_for}s until window resets..."
        )
        time.sleep(sleep_for)


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response | None:
    """
    GET wrapper with automatic throttle inspection and 429 back-off retry.
    Returns the Response on success, or None after exhausting retries.
    """
    if not FOOTBALL_DATA_KEY:
        print("Warning: FOOTBALL_DATA_KEY not set.")
        return None

    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        except requests.RequestException as exc:
            print(f"[fetch_matches] Network error (attempt {attempt}/{retries}): {exc}")
            time.sleep(5 * attempt)
            continue

        if resp.status_code == 429:
            # API explicitly told us to back off.
            retry_after = int(resp.headers.get("Retry-After", 60))
            print(
                f"[fetch_matches] 429 Too Many Requests. "
                f"Retrying after {retry_after}s (attempt {attempt}/{retries})..."
            )
            time.sleep(retry_after + SLEEP_BUFFER_SECONDS)
            continue

        if resp.status_code != 200:
            print(f"[fetch_matches] HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        _throttle(resp)
        return resp

    print("[fetch_matches] All retries exhausted.")
    return None


def fetch_upcoming_matches() -> list:
    """
    Fetches scheduled matches for the 2026 FIFA World Cup (competition ID 2000).
    Returns a list of match dicts, or an empty list on failure.
    """
    url = f"{FOOTBALL_DATA_API_URL}/competitions/2000/matches"
    resp = _get(url, params={"status": "SCHEDULED"})
    if resp is None:
        return []
    return resp.json().get("matches", [])


def fetch_match_by_id(match_id: int) -> dict | None:
    """Fetch a single match by its football-data.org match ID."""
    url = f"{FOOTBALL_DATA_API_URL}/matches/{match_id}"
    resp = _get(url)
    if resp is None:
        return None
    return resp.json()


if __name__ == "__main__":
    matches = fetch_upcoming_matches()
    print(f"Fetched {len(matches)} upcoming matches.")
    if matches:
        first = matches[0]
        print(f"Next match: {first.get('homeTeam', {}).get('name')} vs "
              f"{first.get('awayTeam', {}).get('name')} "
              f"on {first.get('utcDate', 'unknown date')}")
