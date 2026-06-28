"""
Match-context helpers that connect external data feeds (The Odds API) to the
prediction models.

Responsibilities:
  - Normalize team names so SportsPredict / Odds API names line up.
  - Build a lookup of market probabilities keyed by the two teams in a match.
  - Strip the bookmaker margin ("vig") so implied probabilities sum to 1.
  - Estimate per-team expected goals (xG) from the market, which feeds the
    Poisson / player models for markets the bookmaker odds don't cover.

NOTE: xG here is derived from market odds as a pragmatic proxy. If you later add
a real xG feed (Understat, FBref, StatsBomb), replace `estimate_team_xg` with it.
"""

import re
import unicodedata

# Rough average total goals in a World Cup match, used when the totals market
# is unavailable. Used only as a fallback.
DEFAULT_TOTAL_GOALS = 2.6

# How strongly match "supremacy" (favorite vs underdog) skews the goal split.
SUPREMACY_WEIGHT = 0.55

# SportsPredict labels matches with 3-letter FIFA country codes (e.g.
# "GHA vs PAN"), while The Odds API uses full country names ("Ghana").
# This table resolves codes to the full names so the two feeds line up.
FIFA_CODES = {
    "ARG": "Argentina", "AUS": "Australia", "AUT": "Austria", "BEL": "Belgium",
    "BIH": "Bosnia and Herzegovina", "BRA": "Brazil", "CAN": "Canada",
    "CIV": "Ivory Coast", "CMR": "Cameroon", "COL": "Colombia", "CRC": "Costa Rica",
    "CRO": "Croatia", "CZE": "Czechia", "DEN": "Denmark", "ECU": "Ecuador",
    "EGY": "Egypt", "ENG": "England", "ESP": "Spain", "FRA": "France",
    "GER": "Germany", "GHA": "Ghana", "GRE": "Greece", "HAI": "Haiti",
    "IRN": "Iran", "IRQ": "Iraq", "ITA": "Italy", "JOR": "Jordan",
    "JPN": "Japan", "KOR": "South Korea", "KSA": "Saudi Arabia", "MAR": "Morocco",
    "MEX": "Mexico", "NED": "Netherlands", "NGA": "Nigeria", "NOR": "Norway",
    "NZL": "New Zealand", "PAN": "Panama", "PAR": "Paraguay", "PER": "Peru",
    "POL": "Poland", "POR": "Portugal", "QAT": "Qatar", "RSA": "South Africa",
    "SCO": "Scotland", "SEN": "Senegal", "SRB": "Serbia", "SUI": "Switzerland",
    "SWE": "Sweden", "TUN": "Tunisia", "TUR": "Turkiye", "UAE": "United Arab Emirates",
    "URU": "Uruguay", "USA": "USA", "UZB": "Uzbekistan", "VEN": "Venezuela",
    "WAL": "Wales", "CPV": "Cabo Verde", "CUW": "Curacao", "NCL": "New Caledonia",
    "JAM": "Jamaica", "HON": "Honduras", "ALG": "Algeria", "DZA": "Algeria",
    "COD": "DR Congo", "CGO": "Congo", "ANG": "Angola", "ZAM": "Zambia",
}


# Alias table mapping normalized name variants to a single canonical form, so
# the same nation from different feeds (e.g. "Czechia" vs "Czech Republic")
# resolves to one key. Keys and values are already lowercase/accent-stripped.
TEAM_ALIASES = {
    "czechia": "czech republic",
    "turkiye": "turkey",
    "cabo verde": "cape verde",
    "bosnia and herzegovina": "bosnia herzegovina",
    "united states": "usa",
    "united states of america": "usa",
    "korea republic": "south korea",
    "ir iran": "iran",
}


def resolve_team_token(token: str) -> str:
    """
    Map a SportsPredict team token to a full country name.
    A 3-letter FIFA code resolves to the full name; anything else passes through.
    """
    if not token:
        return ""
    code = token.strip().upper()
    if code in FIFA_CODES:
        return FIFA_CODES[code]
    return token.strip()


def normalize_team_name(name: str) -> str:
    """Lowercase, strip accents and punctuation so names match across feeds."""
    if not name:
        return ""
    # Resolve FIFA codes (e.g. "GHA" -> "Ghana") before normalizing.
    name = resolve_team_token(name)
    # Remove accents (e.g. "Türkiye" -> "Turkiye")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    # Drop anything that isn't a letter, digit or space
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Drop a leading article so "the Netherlands" == "Netherlands" and
    # "the United States" resolves through the alias table to "usa".
    name = re.sub(r"^the\s+", "", name)
    # Collapse known spelling variants to a single canonical key.
    return TEAM_ALIASES.get(name, name)


def split_match_name(match_name: str):
    """
    Split "Mexico vs South Africa" -> ("Mexico", "South Africa").
    Returns (home, away) using the original casing. Falls back to (name, "").
    """
    if not match_name:
        return "", ""
    parts = re.split(r"\s+vs\.?\s+|\s+v\s+|\s+-\s+", match_name, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return match_name.strip(), ""


def _implied_no_vig(prices):
    """
    Convert a list of decimal odds into vig-free implied probabilities.
    Returns a list of probabilities summing to 1.0 (or None if input invalid).
    """
    if not prices or any(p is None or p <= 1.0 for p in prices):
        return None
    raw = [1.0 / p for p in prices]
    total = sum(raw)
    if total <= 0:
        return None
    return [r / total for r in raw]


def build_odds_index(odds_list):
    """
    Build a lookup from a frozenset of the two normalized team names to a dict:
        {
          "home_team", "away_team",            # original names from the feed
          "p_home", "p_draw", "p_away",         # vig-free h2h probabilities
          "total_goals"                          # over/under line, or None
        }
    Averages the no-vig probabilities across all bookmakers for stability.
    """
    index = {}
    for event in odds_list or []:
        home = event.get("home_team")
        away = event.get("away_team")
        if not home or not away:
            continue

        h2h_probs = []   # list of (p_home, p_draw, p_away)
        total_lines = []

        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                key = market.get("key")
                outcomes = market.get("outcomes", [])

                if key == "h2h":
                    by_name = {o.get("name"): o.get("price") for o in outcomes}
                    prices = [
                        by_name.get(home),
                        by_name.get("Draw"),
                        by_name.get(away),
                    ]
                    probs = _implied_no_vig(prices)
                    if probs:
                        h2h_probs.append(probs)

                elif key == "totals":
                    for o in outcomes:
                        if o.get("point") is not None:
                            total_lines.append(float(o["point"]))
                            break

        if not h2h_probs:
            continue

        n = len(h2h_probs)
        p_home = sum(p[0] for p in h2h_probs) / n
        p_draw = sum(p[1] for p in h2h_probs) / n
        p_away = sum(p[2] for p in h2h_probs) / n
        total_goals = sum(total_lines) / len(total_lines) if total_lines else None

        key = frozenset({normalize_team_name(home), normalize_team_name(away)})
        index[key] = {
            "home_team": home,
            "away_team": away,
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "total_goals": total_goals,
        }
    return index


def find_match_odds(index, match_name):
    """Look up odds for a SportsPredict match name like 'Mexico vs South Africa'."""
    home, away = split_match_name(match_name)
    key = frozenset({normalize_team_name(home), normalize_team_name(away)})
    return index.get(key)


def estimate_team_xg(p_home, p_draw, p_away, total_goals=None):
    """
    Estimate (xg_home, xg_away) from market probabilities and the totals line.

    The totals line is our estimate of total expected goals; we split it between
    the teams according to how lopsided the win probabilities are.
    """
    total = total_goals if total_goals else DEFAULT_TOTAL_GOALS

    # Supremacy in [-1, 1]: positive favors home.
    supremacy = (p_home or 0.0) - (p_away or 0.0)
    home_share = 0.5 * (1 + supremacy * SUPREMACY_WEIGHT)
    home_share = max(0.15, min(0.85, home_share))

    xg_home = total * home_share
    xg_away = total * (1 - home_share)
    # Keep xG strictly positive so the Poisson model is well-defined.
    return max(0.2, xg_home), max(0.2, xg_away)


def team_is_home(match_name, team_name):
    """Return True if `team_name` is the home side of `match_name`."""
    home, _ = split_match_name(match_name)
    return normalize_team_name(home) == normalize_team_name(team_name)


if __name__ == "__main__":
    sample = [
        {
            "home_team": "Mexico",
            "away_team": "South Africa",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Mexico", "price": 1.8},
                                {"name": "Draw", "price": 3.5},
                                {"name": "South Africa", "price": 4.5},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.9, "point": 2.5},
                                {"name": "Under", "price": 1.9, "point": 2.5},
                            ],
                        },
                    ]
                }
            ],
        }
    ]
    idx = build_odds_index(sample)
    odds = find_match_odds(idx, "Mexico vs South Africa")
    print("Odds:", odds)
    print("xG:", estimate_team_xg(odds["p_home"], odds["p_draw"], odds["p_away"], odds["total_goals"]))
