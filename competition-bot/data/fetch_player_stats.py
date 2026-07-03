"""
Fetches per-player stats from api-football.com (v3), with strict quota safety.

Free-tier constraints (confirmed against the live API):
  - 10 requests/minute, 100 requests/day
  - The /players `search` param REQUIRES a `league` or `team` param; name-only
    search is rejected. We therefore search within the World Cup (league=1,
    season=2026). Coverage on the free plan is limited, so misses are common.

To stay inside the quota we:
  - Persist results to a JSON disk cache so a player is fetched at most once,
    ever (across runs).
  - Read the rate-limit headers and sleep when the per-minute window is nearly
    exhausted.
  - Trip a circuit breaker on 429 / daily-limit exhaustion so we stop calling
    the API for the rest of the run and serve fallbacks instead.

When real data isn't available we return FALLBACK_STATS, calibrated to a typical
"featured" attacker (the kind of player these markets are usually about), so the
downstream probability is a sensible prior rather than a coin flip.
"""

import os
import json
import time
import unicodedata
import requests
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"

# World Cup identifiers in api-football.
WC_LEAGUE_ID = 1
WC_SEASON = 2026

# Disk cache lives next to this file.
_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".player_cache.json")

MIN_MINUTES = 180

# Calibrated to a typical featured attacker. Tuned so that:
#   P(1+ shot on target, full match) = 1 - e^(-0.85) ≈ 0.57
#   P(goal)        = 1 - e^(-(2.2 * 0.13)) ≈ 0.25
#   P(assist)      = 1 - e^(-0.18)         ≈ 0.16
FALLBACK_STATS = {
    "shots_per_90": 2.2,
    "shots_on_target_per_90": 0.85,
    "conversion_rate": 0.13,
    "xA_per_90": 0.18,
    "is_real": False,
}

# Module state.
_cache = None            # loaded lazily from disk
_api_disabled = False    # circuit breaker for the current run


def _headers():
    return {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_FOOTBALL_KEY or "",
    }


def _normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
                _cache = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is not None:
        try:
            with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
                json.dump(_cache, fh, indent=2)
        except OSError as exc:
            print(f"[fetch_player_stats] Could not write cache: {exc}")


def peek_cache(player_name: str):
    """
    Return the cached stats dict for a player WITHOUT any network call, or None
    if the player isn't cached yet. Lets callers tell a cache hit (free, no
    quota) from a player that would require a live lookup.
    """
    return _load_cache().get(_normalize(player_name))


def _respect_rate_limits(resp: requests.Response) -> None:
    """Sleep if the per-minute window is nearly used up; trip breaker on daily."""
    global _api_disabled
    try:
        per_min_remaining = int(resp.headers.get("x-ratelimit-remaining", 99))
        per_day_remaining = int(resp.headers.get("x-ratelimit-requests-remaining", 99))
    except (TypeError, ValueError):
        return

    if per_day_remaining <= 1:
        print("[fetch_player_stats] Daily API quota exhausted; using fallbacks for the rest of the run.")
        _api_disabled = True
        return

    if per_min_remaining <= 1:
        print("[fetch_player_stats] Per-minute quota low; sleeping 60s...")
        time.sleep(60)


def _safe_div(num, den, default=0.0):
    try:
        if den and float(den) > 0:
            return float(num or 0) / float(den)
    except (TypeError, ValueError):
        pass
    return default


def _parse_stats(stats_list: list):
    best, best_minutes = None, 0
    for entry in stats_list or []:
        minutes = (entry.get("games") or {}).get("minutes") or 0
        if minutes > best_minutes:
            best_minutes, best = minutes, entry
    if best is None or best_minutes < MIN_MINUTES:
        return None

    games = best.get("games") or {}
    shots = best.get("shots") or {}
    goals = best.get("goals") or {}
    nineties = (float(games.get("minutes") or 0) / 90.0) or 1.0

    total_shots = float(shots.get("total") or 0)
    return {
        "shots_per_90": round(total_shots / nineties, 3),
        "shots_on_target_per_90": round(float(shots.get("on") or 0) / nineties, 3),
        "conversion_rate": round(_safe_div(goals.get("total"), total_shots, 0.10), 3),
        "xA_per_90": round(float(goals.get("assists") or 0) / nineties, 3),
        "is_real": True,
    }


def _search_player(name: str):
    global _api_disabled
    if not API_FOOTBALL_KEY or _api_disabled:
        return None

    try:
        resp = requests.get(
            f"{BASE_URL}/players",
            headers=_headers(),
            params={"search": name, "league": WC_LEAGUE_ID, "season": WC_SEASON},
            timeout=5,   # short timeout — don't block the main loop
        )
    except requests.RequestException as exc:
        print(f"[fetch_player_stats] Request error for '{name}': {exc}")
        return None

    if resp.status_code == 429:
        print("[fetch_player_stats] 429 rate limited; disabling API for this run.")
        _api_disabled = True
        return None
    if resp.status_code != 200:
        print(f"[fetch_player_stats] HTTP {resp.status_code} for '{name}'")
        return None

    _respect_rate_limits(resp)

    players = (resp.json() or {}).get("response") or []
    if not players:
        return None
    return _parse_stats(players[0].get("statistics") or [])


def fetch_player_stats(player_name: str) -> dict:
    """
    Return a stats dict for a player:
        player_name, shots_per_90, shots_on_target_per_90,
        conversion_rate, xA_per_90, is_real

    `is_real` is False when the numbers come from FALLBACK_STATS. Results are
    cached to disk so each player costs at most one API request ever.
    """
    cache = _load_cache()
    key = _normalize(player_name)

    if key not in cache:
        # Guard: silently skip obvious non-person strings that slip through
        # due to parser mis-routes (team names, articles, common words).
        _NON_PERSON_KEYS = {
            "a substitute", "both teams", "dr congo", "haiti", "curacao",
            "new zealand", "south africa", "ivory coast", "cape verde",
            "burkina faso", "saudi arabia", "south korea", "north korea",
            "costa rica", "el salvador", "trinidad and tobago",
            "bosnia and herzegovina", "united states", "new caledonia",
        }
        if key in _NON_PERSON_KEYS:
            import logging
            logging.getLogger(__name__).warning(
                "[fetch_player_stats] Blocked cache write for non-person key %r", key
            )
            result = FALLBACK_STATS.copy()
            result["player_name"] = player_name
            return result
        stats = _search_player(player_name)
        if stats is not None:
            # Real data found — cache it permanently.
            cache[key] = stats
            _save_cache()
        elif _api_disabled:
            # The API was rate-limited / disabled this run, so we don't actually
            # know if this player exists. Return a fallback WITHOUT caching, so
            # a future run (after quota resets) can try again.
            result = FALLBACK_STATS.copy()
            result["player_name"] = player_name
            return result
        else:
            # The API responded but had no usable data for this player. Cache
            # this "miss" so we never waste another request on them.
            miss = FALLBACK_STATS.copy()
            cache[key] = miss
            _save_cache()

    result = cache[key].copy()
    result["player_name"] = player_name
    return result


if __name__ == "__main__":
    for name in ["Kylian Mbappe", "Patrik Schick", "Nobody XYZ"]:
        s = fetch_player_stats(name)
        flag = "real" if s.get("is_real") else "FALLBACK"
        print(f"{name:18} [{flag}]  SOT/90={s['shots_on_target_per_90']}  "
              f"shots/90={s['shots_per_90']}  conv={s['conversion_rate']}  xA/90={s['xA_per_90']}")
