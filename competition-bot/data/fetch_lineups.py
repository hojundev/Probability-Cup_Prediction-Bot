"""
Fetches confirmed starting lineups from api-football.com (v3).

Confirmed lineups are released ~60 minutes before kickoff. Knowing the starting
XI lets the bot slash player-market probabilities for anyone who is benched or
absent (a benched star is a big edge over the crowd, who anchor on reputation).

Quota safety mirrors fetch_player_stats: this shares the same 100 req/day,
10 req/min api-football budget, so callers should only poll inside the
pre-kickoff window and stop once a lineup has been consumed.

Endpoint:
    GET /fixtures/lineups?fixture={fixture_id}

Response shape (trimmed):
    {
      "response": [
        {
          "team": {"id": 9, "name": "Mexico"},
          "startXI": [
            {"player": {"id": 1, "name": "Guillermo Ochoa", "pos": "G"}},
            ...
          ],
          "substitutes": [ {"player": {...}}, ... ]
        },
        ... (one entry per team)
      ]
    }

An empty `response` array means lineups are not published yet.
"""

import os
import json
import unicodedata

import requests
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"

# World Cup identifiers in api-football (match fetch_player_stats).
WC_LEAGUE_ID = 1
WC_SEASON = 2026

# Disk cache for the fixture list. Fixtures (ids, dates, teams) are static for
# the tournament, so we fetch them at most once and reuse across runs.
_FIXTURE_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".fixture_cache.json")
_fixture_index = None  # lazily built {frozenset({norm_home, norm_away}): fixture_id}


class LineupRateLimited(Exception):
    """Raised when api-football returns HTTP 429 during a lineup check."""


def _headers():
    return {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_FOOTBALL_KEY or "",
    }


def normalize_player_name(name: str) -> str:
    """Lowercase + strip accents so lineup names match question names."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.lower().strip()


def fetch_lineup(fixture_id):
    """
    Return the set of normalized starting-XI player names for a fixture, or
    None if lineups are not yet available (empty response / no key).

    Raises LineupRateLimited on HTTP 429 so the scheduler can skip the cycle.
    """
    if not API_FOOTBALL_KEY or fixture_id is None:
        return None

    try:
        resp = requests.get(
            f"{BASE_URL}/fixtures/lineups",
            headers=_headers(),
            params={"fixture": fixture_id},
            timeout=5,
        )
    except requests.RequestException as exc:
        print(f"[fetch_lineups] Request error for fixture {fixture_id}: {exc}")
        return None

    if resp.status_code == 429:
        raise LineupRateLimited(f"429 for fixture {fixture_id}")
    if resp.status_code != 200:
        print(f"[fetch_lineups] HTTP {resp.status_code} for fixture {fixture_id}")
        return None

    response = (resp.json() or {}).get("response") or []
    if not response:
        return None

    starters = set()
    for team_block in response:
        for entry in team_block.get("startXI") or []:
            player = (entry.get("player") or {})
            name = player.get("name")
            if name:
                starters.add(normalize_player_name(name))
    return starters or None


def player_is_starter(player_name, starters):
    """
    Return True if `player_name` appears in the confirmed starting XI.

    Matches on normalized full name, and also on a last-name fallback so that
    "Schick" matches "Patrik Schick".
    """
    if not starters:
        return None  # no lineup data -> caller should leave prediction unchanged
    norm = normalize_player_name(player_name)
    if norm in starters:
        return True
    # Last-name fallback (questions sometimes use surname only, or vice versa).
    last = norm.split()[-1] if norm else ""
    for s in starters:
        if last and (last == s.split()[-1] or last in s.split()):
            return True
    return False


# ---------------------------------------------------------------------------
# Fixture resolution: SportsPredict match name -> api-football fixture id.
#
# SportsPredict matches carry no api-football fixture id, so we fetch the full
# World Cup fixture list once (cached to disk) and match on normalized team
# names. This is what lets us call fetch_lineup() for a given match.
# ---------------------------------------------------------------------------

def _load_fixture_cache():
    try:
        with open(_FIXTURE_CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_fixture_cache(fixtures):
    try:
        with open(_FIXTURE_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(fixtures, fh, indent=2)
    except OSError as exc:
        print(f"[fetch_lineups] Could not write fixture cache: {exc}")


def fetch_wc_fixtures(force=False):
    """
    Return a list of {"fixture_id", "home", "away"} for the World Cup, fetched
    from api-football's /fixtures endpoint. Cached to disk so it costs at most
    one API request for the whole tournament. Returns [] if unavailable.

    Raises LineupRateLimited on HTTP 429 so the caller can skip the cycle.
    """
    if not force:
        cached = _load_fixture_cache()
        if cached:
            return cached

    if not API_FOOTBALL_KEY:
        return []

    try:
        resp = requests.get(
            f"{BASE_URL}/fixtures",
            headers=_headers(),
            params={"league": WC_LEAGUE_ID, "season": WC_SEASON},
            timeout=8,
        )
    except requests.RequestException as exc:
        print(f"[fetch_lineups] Fixture request error: {exc}")
        return []

    if resp.status_code == 429:
        raise LineupRateLimited("429 fetching fixtures")
    if resp.status_code != 200:
        print(f"[fetch_lineups] HTTP {resp.status_code} fetching fixtures")
        return []

    fixtures = []
    for item in (resp.json() or {}).get("response") or []:
        fid = (item.get("fixture") or {}).get("id")
        teams = item.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        if fid and home and away:
            fixtures.append({"fixture_id": fid, "home": home, "away": away})

    if fixtures:
        _save_fixture_cache(fixtures)
    return fixtures


def _build_fixture_index(fixtures):
    """Map frozenset({norm_home, norm_away}) -> fixture_id."""
    from bot.match_data import normalize_team_name
    index = {}
    for fx in fixtures or []:
        key = frozenset({normalize_team_name(fx["home"]), normalize_team_name(fx["away"])})
        index[key] = fx["fixture_id"]
    return index


def resolve_fixture_id(match_name, force_refresh=False):
    """
    Resolve a SportsPredict match name ("Mexico vs South Africa") to an
    api-football fixture id, or None if it can't be matched.
    """
    global _fixture_index
    from bot.match_data import split_match_name, normalize_team_name

    if _fixture_index is None or force_refresh:
        _fixture_index = _build_fixture_index(fetch_wc_fixtures(force=force_refresh))

    home, away = split_match_name(match_name)
    key = frozenset({normalize_team_name(home), normalize_team_name(away)})
    return _fixture_index.get(key)


if __name__ == "__main__":
    # Smoke test (requires a valid fixture id + API key).
    import sys
    fid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(fetch_lineup(fid))
