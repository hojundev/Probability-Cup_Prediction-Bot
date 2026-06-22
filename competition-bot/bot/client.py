"""
SportsPredict API client.

Every outbound request goes through SportsPredictClient._request, which:
  1. Acquires a token from the shared TokenBucket (≤55 req/60s by default).
  2. Sets Authorization: Bearer <key> on the request.
  3. Applies a retry policy:
       429  → honour X-RateLimit-Reset header, exponential back-off, retry
       401  → raise AuthError immediately (no retry)
       500  → retry up to max_retries, then raise
       2xx  → return (json_body, response_headers)

Module-level functions are kept as thin wrappers so submit.py keeps working
without changes.
"""

import os
import time
import logging

import requests
from dotenv import load_dotenv

from bot.rate_limiter import TokenBucket

load_dotenv()

log = logging.getLogger(__name__)

API_URL = "https://api.sportspredict.com/api/v1"

# Shared default client – module-level functions delegate here.
_default_client: "SportsPredictClient | None" = None


class AuthError(Exception):
    """Raised on HTTP 401 – bad or missing API key."""


class SportsPredictClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = API_URL,
        session: requests.Session | None = None,
        bucket: TokenBucket | None = None,
        max_retries: int = 4,
        timeout: int = 10,
    ):
        self._key = api_key or os.getenv("SPORTSPREDICT_KEY") or ""
        self._base = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._bucket = bucket or TokenBucket()
        self._max_retries = max_retries
        self._timeout = timeout

        if not self._key:
            log.warning("SPORTSPREDICT_KEY is not set.")

    # ------------------------------------------------------------------
    # Core request dispatcher
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, *, raw: bool = False, **kwargs):
        """
        Issue an authenticated, rate-limited request.

        Returns (parsed_json, headers) tuple so callers that need headers
        (e.g. fetch_markets) can inspect them.  Pass raw=True to skip JSON
        parsing and return the Response object directly.
        """
        url = f"{self._base}/{path.lstrip('/')}"
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._key}"

        backoff = 2.0
        server_error_attempts = 0
        while True:
            self._bucket.acquire()
            try:
                resp = self._session.request(
                    method, url, headers=headers, timeout=self._timeout, **kwargs
                )
            except requests.RequestException as exc:
                log.error("Request error on %s %s: %s", method, path, exc)
                server_error_attempts += 1
                if server_error_attempts <= self._max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    continue
                raise

            if resp.status_code == 401:
                raise AuthError(
                    f"HTTP 401 on {method} {path} – check SPORTSPREDICT_KEY"
                )

            if resp.status_code == 429:
                # 429 retries are unlimited — just wait for the window to reset
                reset = int(resp.headers.get("X-RateLimit-Reset", 60))
                wait = max(reset + 2, backoff)
                log.warning("429 rate-limited on %s %s, sleeping %ss",
                            method, path, wait)
                time.sleep(wait)
                backoff = min(backoff * 2, 120)
                continue  # don't count against server_error_attempts

            if resp.status_code >= 500:
                server_error_attempts += 1
                log.warning("HTTP %s on %s %s (server error attempt %d/%d)",
                            resp.status_code, method, path,
                            server_error_attempts, self._max_retries)
                if server_error_attempts <= self._max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    continue
                resp.raise_for_status()

            if raw:
                return resp

            if not resp.ok:
                resp.raise_for_status()

            try:
                return resp.json(), resp.headers
            except ValueError:
                return {}, resp.headers

    # ------------------------------------------------------------------
    # Endpoint methods
    # ------------------------------------------------------------------

    def fetch_events(self):
        body, _ = self._request("GET", "/events")
        return body

    def fetch_lobbies(self, event_id: str):
        body, _ = self._request("GET", "/lobbies", params={"event_id": event_id})
        return body

    def join_lobby(self, lobby_id: str):
        """409 means already joined – that's fine."""
        resp = self._request("POST", f"/lobbies/{lobby_id}/join", raw=True)
        if resp.status_code == 409:
            return resp
        if not resp.ok:
            resp.raise_for_status()
        return resp

    def fetch_matches(self, event_id: str, lobby_id: str | None = None):
        params = {"event_id": event_id}
        if lobby_id:
            params["lobby_id"] = lobby_id
        body, _ = self._request("GET", "/matches", params=params)
        return body

    def fetch_markets(self, lobby_id: str, match_id: str):
        """Returns (markets_list, headers)."""
        body, hdrs = self._request(
            "GET", "/markets",
            params={"lobby_id": lobby_id, "match_id": match_id},
        )
        return body, hdrs

    def fetch_my_predictions(self, lobby_id: str | None = None):
        params = {}
        if lobby_id:
            params["lobby_id"] = lobby_id
        body, _ = self._request("GET", "/predictions", params=params)
        return body

    def update_prediction(self, prediction_id: str, probability: int):
        body, _ = self._request(
            "PATCH", f"/predictions/{prediction_id}",
            json={"probability": probability},
        )
        return body

    def submit_predictions_batch(self, predictions: list[dict]):
        """
        POST up to 50 new predictions.
        Returns the full batch response including per-item success/failure.
        If a prediction returns 409, the caller should PATCH instead.
        """
        assert len(predictions) <= 50, "Batch must be ≤ 50 predictions"
        body, _ = self._request(
            "POST", "/predictions/batch",
            json={"predictions": predictions},
        )
        failed = [r for r in body.get("results", []) if not r.get("success")]
        for f in failed:
            log.warning("Batch prediction failed for market %s: %s",
                        f.get("market_id"), f.get("error"))
        return body

    def get_probability_cup_lobby_and_event(self):
        """
        Locate the Probability Cup event and lobby.
        Matches by title (live API returns type as a UUID, not 'probability').
        """
        events = self.fetch_events()
        if not events:
            raise ValueError("No events returned by the API.")

        event = next(
            (e for e in events if "probability cup" in (e.get("title") or "").lower()),
            None,
        )
        if not event and len(events) == 1:
            event = events[0]
        if not event:
            raise ValueError("No Probability Cup event found.")

        lobbies = self.fetch_lobbies(event["id"])
        if not lobbies:
            raise ValueError("No lobbies found for Probability Cup event.")

        return event, lobbies[0]


# ------------------------------------------------------------------
# Module-level wrappers (backward-compat for submit.py imports)
# ------------------------------------------------------------------

def _client() -> SportsPredictClient:
    global _default_client
    if _default_client is None:
        _default_client = SportsPredictClient()
    return _default_client


def fetch_events():
    return _client().fetch_events()

def fetch_lobbies(event_id):
    return _client().fetch_lobbies(event_id)

def join_lobby(lobby_id):
    return _client().join_lobby(lobby_id)

def fetch_matches(event_id, lobby_id=None):
    return _client().fetch_matches(event_id, lobby_id)

def fetch_markets(lobby_id, match_id):
    return _client().fetch_markets(lobby_id, match_id)

def fetch_my_predictions(lobby_id=None):
    return _client().fetch_my_predictions(lobby_id)

def update_prediction(prediction_id, probability):
    return _client().update_prediction(prediction_id, probability)

def submit_predictions_batch(predictions):
    return _client().submit_predictions_batch(predictions)

def get_probability_cup_lobby_and_event():
    return _client().get_probability_cup_lobby_and_event()
