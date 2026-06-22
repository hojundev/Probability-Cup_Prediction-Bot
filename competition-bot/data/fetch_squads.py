"""
Player -> national-team resolver for the 2026 World Cup.

Player prop questions ("Will Salem Al-Dawsari score?") don't name a team, and
in lopsided matches the two teams' xG differ wildly — so we must know which side
the player is on to price the market correctly.

This module builds a {normalized_player_name: normalized_team_name} map from
api-football squads ONCE and caches it to disk (`.squad_cache.json`). The build
costs ~1 + (number of WC teams) api-football requests, so it is an explicit,
one-time step the operator runs:

    python3 -m data.fetch_squads          # build/refresh the squad cache

During normal submission runs, `resolve_player_team()` only READS the cache and
never makes API calls — so it can never surprise-burn the daily quota.

Endpoints used (api-football v3):
    GET /teams?league=1&season=2026             -> WC team ids + names
    GET /players/squads?team={team_id}          -> squad list per team
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

WC_LEAGUE_ID = 1
WC_SEASON = 2026

# Season used to DISCOVER national-team ids when building the squad map. The
# free api-football plan only serves seasons 2022-2024, and the World Cup
# (league 1) only ran in 2022 within that range. National-team membership is
# stable across tournaments, so the 2022 WC team list is a sound basis: we use
# its team ids to pull each nation's CURRENT squad via /players/squads.
# Limitation: only the 32 teams from WC 2022 are discoverable, so players on the
# 16 new 2026 qualifiers fall back to the conservative base rate.
SQUAD_DISCOVERY_SEASON = 2022

# Search-term aliases for nations whose api-football name differs from the
# SportsPredict/FIFA name. Keys are normalized (lowercase, no punctuation);
# values are the term to send to /teams?search.
SEARCH_ALIASES = {
    "cabo verde": "Cape Verde",
    "dr congo": "Congo DR",
    "bosnia and herzegovina": "Bosnia",
    "south korea": "South Korea",
    "ivory coast": "Ivory Coast",
    "turkiye": "Turkey",
}

_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".squad_cache.json")

# Lazily loaded { norm_player_name: norm_team_name }.
_squad_index = None
# Last-name fallback index: { norm_last_name: set(norm_team_name) }.
_lastname_index = None


def _headers():
    return {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_FOOTBALL_KEY or "",
    }


def _norm(name: str) -> str:
    """Lowercase, strip accents, drop punctuation (incl. hyphens), collapse spaces."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = "".join(c if c.isalnum() or c.isspace() else " " for c in name)
    return " ".join(name.split())


# ---------------------------------------------------------------------------
# Build (explicit, quota-spending) — run once via `python3 -m data.fetch_squads`
# ---------------------------------------------------------------------------

def _get(path, params):
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=8)
    if resp.status_code == 429:
        raise RuntimeError("api-football 429 rate limited during squad build")
    if resp.status_code != 200:
        raise RuntimeError(f"api-football HTTP {resp.status_code} on {path}")
    # Be polite to the per-minute window.
    try:
        if int(resp.headers.get("x-ratelimit-remaining", 99)) <= 1:
            print("[fetch_squads] per-minute quota low; sleeping 60s...")
            time.sleep(60)
    except (TypeError, ValueError):
        pass
    return resp.json() or {}


def build_squad_cache():
    """
    Fetch all WC team squads and write {player -> team} to the disk cache.
    Returns the built index. Raises on missing key or hard API failure.
    """
    if not API_FOOTBALL_KEY:
        raise RuntimeError("API_FOOTBALL_KEY not set; cannot build squad cache.")

    teams_body = _get("/teams", {"league": WC_LEAGUE_ID, "season": SQUAD_DISCOVERY_SEASON})
    teams = teams_body.get("response") or []
    if not teams:
        err = teams_body.get("errors")
        raise RuntimeError(
            f"No WC teams returned for season {SQUAD_DISCOVERY_SEASON} "
            f"(api errors: {err}); cannot build squad cache."
        )

    index = {}
    for entry in teams:
        team = entry.get("team") or {}
        team_id = team.get("id")
        team_name = team.get("name")
        if not team_id or not team_name:
            continue
        squad_body = _get("/players/squads", {"team": team_id})
        for block in squad_body.get("response") or []:
            for player in block.get("players") or []:
                pname = player.get("name")
                if pname:
                    index[_norm(pname)] = _norm(team_name)
        print(f"[fetch_squads] {team_name}: cached squad.")

    with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"[fetch_squads] Wrote {len(index)} players to {_CACHE_PATH}")
    return index


# ---------------------------------------------------------------------------
# Read-only resolution (used during submission; never calls the API)
# ---------------------------------------------------------------------------

def _load_index():
    global _squad_index, _lastname_index
    if _squad_index is None:
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
                _squad_index = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            _squad_index = {}
        _lastname_index = {}
        for pname, team in _squad_index.items():
            parts = pname.split()
            if parts:
                _lastname_index.setdefault(parts[-1], set()).add(team)
    return _squad_index


def resolve_player_team(player_name):
    """
    Return the normalized national-team name for a player, or None if unknown.
    Read-only: uses the disk cache and never makes an API call.
    """
    index = _load_index()
    if not index:
        return None
    key = _norm(player_name)
    if key in index:
        return index[key]
    # Last-name fallback, only if it maps to exactly one team (unambiguous).
    parts = key.split()
    if parts:
        teams = _lastname_index.get(parts[-1])
        if teams and len(teams) == 1:
            return next(iter(teams))
    return None


# ---------------------------------------------------------------------------
# Gap-fill: add nations that weren't in the 2022 discovery (by name lookup)
# ---------------------------------------------------------------------------

def _search_national_team_id(search_name):
    """
    Resolve a country name to its api-football national-team id via /teams?search.
    Returns (team_id, api_team_name) or (None, None).

    Only MEN'S national teams: women's national teams (name suffix " W") are
    excluded. Some nations need a search alias to match api-football's spelling.
    """
    term = SEARCH_ALIASES.get(_norm(search_name), search_name)
    body = _get("/teams", {"search": term})
    target = _norm(search_name)
    target_alias = _norm(term)

    def _is_womens(name):
        n = _norm(name)
        return n.endswith(" w") or "women" in n

    best = None
    for entry in body.get("response") or []:
        team = entry.get("team") or {}
        if not team.get("national"):
            continue
        name = team.get("name") or ""
        if _is_womens(name):
            continue                      # never pick a women's team
        norm_name = _norm(name)
        if norm_name == target or norm_name == target_alias:
            return team.get("id"), name   # exact men's match
        if best is None:
            best = (team.get("id"), name)  # first men's national team
    return best if best else (None, None)


def fill_squads(country_names, verbose=True):
    """
    Add the given nations' current squads to the existing cache, by name lookup.
    Skips nations already covered. Merges into and rewrites the cache.
    Returns the number of players added.
    """
    index = _load_index()
    covered = set(index.values())
    added = 0
    for name in country_names:
        norm = _norm(name)
        if norm in covered:
            if verbose:
                print(f"[fetch_squads] {name}: already covered, skipping.")
            continue
        try:
            team_id, api_name = _search_national_team_id(name)
        except RuntimeError as exc:
            print(f"[fetch_squads] {name}: lookup failed ({exc}); stopping.")
            break
        if not team_id:
            print(f"[fetch_squads] {name}: no national team found, skipping.")
            continue
        try:
            squad_body = _get("/players/squads", {"team": team_id})
        except RuntimeError as exc:
            print(f"[fetch_squads] {name}: squad fetch failed ({exc}); stopping.")
            break
        team_key = _norm(api_name)
        n = 0
        for block in squad_body.get("response") or []:
            for player in block.get("players") or []:
                pname = player.get("name")
                if pname:
                    index[_norm(pname)] = team_key
                    n += 1
        added += n
        if verbose:
            print(f"[fetch_squads] {name} ({api_name}): added {n} players.")

    if added:
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)
        # Reset in-memory indexes so the new data is picked up.
        global _squad_index, _lastname_index
        _squad_index = None
        _lastname_index = None
        print(f"[fetch_squads] Added {added} players; cache now has {len(index)} total.")
    else:
        print("[fetch_squads] No new players added.")
    return added


def competition_team_names():
    """
    Return the set of full country names appearing in the current SportsPredict
    matches (resolved from the 3-letter tokens). Used to know exactly which
    nations the bot needs squads for.
    """
    from bot.client import get_probability_cup_lobby_and_event, fetch_matches
    from bot.match_data import split_match_name, resolve_team_token

    event, lobby = get_probability_cup_lobby_and_event()
    matches = fetch_matches(event["id"], lobby["id"])
    names = set()
    for m in matches:
        home, away = split_match_name(m.get("name", ""))
        for tok in (home, away):
            full = resolve_team_token(tok)
            if full:
                names.add(full)
    return names


def fill_missing_from_competition(verbose=True):
    """
    Fill squads for exactly the nations that appear in the live competition but
    aren't yet in the cache. Targets only what the bot needs.
    """
    index = _load_index()
    covered = set(index.values())
    needed = competition_team_names()
    missing = [n for n in sorted(needed) if _norm(n) not in covered]
    if verbose:
        print(f"[fetch_squads] {len(needed)} nations in competition, "
              f"{len(missing)} missing: {missing}")
    if not missing:
        print("[fetch_squads] Nothing missing; cache already covers all teams.")
        return 0
    return fill_squads(missing, verbose=verbose)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fill":
        # Fill only the nations the competition needs that aren't cached yet.
        fill_missing_from_competition()
    elif len(sys.argv) > 2 and sys.argv[1] == "add":
        # Manually add specific nations: python3 -m data.fetch_squads add Egypt "New Zealand"
        fill_squads(sys.argv[2:])
    else:
        # One-time full build / refresh from the 2022 WC field.
        build_squad_cache()
