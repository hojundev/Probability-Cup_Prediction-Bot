"""
Parses SportsPredict market questions into structured dicts the model can route.

The questions follow a fairly consistent template language. Each parse result
has a "type" plus the fields that type needs. Anything we can't confidently
classify returns {"type": "unknown"} so the caller can apply a neutral prior.

Ordering matters: more specific patterns are checked before generic ones.

Knockout rounds (Round of 32 onward) qualify almost every question with a time
scope like "in regulation (90 minutes + stoppage time)" and add a batch of new
market types (advancement, clean sheet, draw, BTTS, scoring N+ goals, time-window
goal markets, etc.). `_strip_scope` removes the qualifier up front so the
group-stage patterns keep matching, and the extra knockout types are handled
explicitly below.
"""

import re

# Connectors used to split "Will <team> <verb> ..." style questions.
_TEAM_RE = r"(.+?)"


def _threshold(q):
    """
    Extract (n, direction) from a question.
      "3 or more"  -> (3, "over")     meaning >= 3
      "2 or fewer" -> (2, "under")    meaning <= 2
      "at least 1" -> (1, "over")
    Returns (None, None) if no threshold is present.
    """
    m = re.search(r"(\d+)\s+or\s+more", q)
    if m:
        return int(m.group(1)), "over"
    m = re.search(r"(\d+)\s+or\s+fewer", q)
    if m:
        return int(m.group(1)), "under"
    m = re.search(r"at\s+least\s+(\d+)", q)
    if m:
        return int(m.group(1)), "over"
    return None, None


def _half(q):
    """Return 'first', 'second', or None based on the half mentioned."""
    if "first half" in q:
        return "first"
    if "second half" in q:
        return "second"
    return None


def _title(raw):
    """Title-case a captured team token (or 'Unknown')."""
    return raw.strip().title() if raw and raw.strip() else "Unknown"


def _clean_player_name(raw):
    """
    Title-case a captured player name and drop a trailing '(Country)' annotation.

    Knockout player markets tag the player with their nation, e.g.
    "Lionel Messi (Argentina)". That annotation breaks squad/stat lookups, so we
    strip it and keep just the name.
    """
    if not raw:
        return "Unknown"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", raw).strip()
    return name.title() if name else "Unknown"


# Geographic qualifiers that appear as the FIRST word of national team names
# but never as the first word of a player's given/surname. Used to distinguish
# "DR Congo" (team) from "Yoane Wissa" (player) when there is no "(Country)" tag.
_GEO_QUALIFIERS = frozenset({
    "dr", "new", "south", "north", "united", "el", "trinidad",
    "bosnia", "ivory", "cape", "saudi", "costa", "central",
    "equatorial", "burkina", "sierra", "guinea",
})


def _looks_like_team(subject: str) -> bool:
    """
    Return True if `subject` looks like a national team name rather than a
    player name. Used to avoid routing team-goal/SOT markets to player handlers.

    Two signals:
    1. The first word is a known geographic qualifier (DR, New, South, United,
       …). Player names never start with these. Catches "DR Congo", "New
       Zealand", "South Africa", etc.
    2. The subject is a single word AND it normalizes to a known FIFA country
       name. Catches "Spain", "Germany", "France", etc. — single-word team
       names that don't have a geo-qualifier but are clearly not player surnames
       when they appear as the sole subject of a "have N or more SOT" question.
    """
    if not subject:
        return False
    has_country_tag = bool(re.search(r"\([^)]+\)", subject))
    if has_country_tag:
        return False   # "(Country)" tag = definitively a player

    from bot.match_data import normalize_team_name, FIFA_CODES
    # Signal 1: the normalized subject exactly matches a known FIFA country name.
    # normalize_team_name handles the leading "the", accents, and aliases, so
    # "the United States" -> "usa", "Bosnia and Herzegovina" -> canonical, etc.
    norm = normalize_team_name(subject)
    known_countries = {normalize_team_name(v) for v in FIFA_CODES.values()}
    if norm in known_countries:
        return True
    # Signal 2: first word is a geographic qualifier (covers any country not in
    # FIFA_CODES, e.g. a team whose full name we don't have mapped).
    if subject.strip().split()[0].lower() in _GEO_QUALIFIERS:
        return True
    return False


def _strip_scope(q):
    """
    Remove knockout time-scope qualifiers that don't change how we model a market
    (our xG is a 90-minute line). Keeps "(excluding own goals)" — a real player
    signal — and keeps a bare "regulation" subject (e.g. "will regulation end in
    a tie?").
    """
    q = q.replace("(90 minutes + stoppage time)", " ")
    q = q.replace("(90 minutes and stoppage time)", " ")
    q = q.replace(", excluding extra time", " ")
    q = q.replace(", including any extra time", " ")
    q = q.replace("including any extra time", " ")
    # "in/of/during regulation" -> drop (but not a leading "regulation" subject).
    q = re.sub(r"\b(?:in|of|during)\s+regulation\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def parse_question(question: str) -> dict:
    q = _strip_scope(question.lower().strip())

    # ---- Compound questions we explicitly model ----------------------------
    # "Will both teams score AND the match have 3 or more total goals?"
    if "both teams score" in q and "total goals" in q:
        n, direction = _threshold(q)
        return {"type": "btts_and_total_goals", "threshold": n or 3, "direction": direction or "over"}

    # "Will a penalty kick be awarded OR a red card be shown ...?"
    if "penalty kick" in q and "red card" in q:
        return {"type": "penalty_or_red_card"}
    # "Will a penalty kick be scored?" — distinct from "awarded": awarded AND
    # converted. Checked before the plain awarded/shootout branches.
    if "penalty" in q and ("be scored" in q or "be converted" in q) and "shootout" not in q:
        return {"type": "penalty_scored"}
    if "penalty kick be awarded" in q:
        return {"type": "penalty_awarded"}

    # "Will both teams score ...?" (plain BTTS). After the BTTS+totals compound,
    # and excluding the "both teams receive a card" market.
    if "both teams score" in q and "card" not in q:
        return {"type": "btts"}

    # Other AND/OR compounds we don't model precisely -> unknown (neutral prior)
    if (" and " in q or " or " in q) and "shot on target" not in q and "score or assist" not in q:
        if "score the first goal" in q:
            return {"type": "unknown", "raw": question}

    # ---- Player markets ----------------------------------------------------
    # "Will any <team> player score/assist...?" — team-level market, not a single
    # player. Checked first so "any Portugal player" isn't captured as a name.
    if "any" in q and "player" in q and "score" in q and (
        "more than 1 goal" not in q and "2 or more goals" not in q and "brace" not in q):
        m = re.search(r"will any\s+" + _TEAM_RE + r"\s+player", q)
        if m and _looks_like_team(m.group(1)):
            return {"type": "team_score", "team": _title(m.group(1))}

    # "Will <player> score or assist a goal (excluding own goals)?"
    if "score or assist" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score or assist", q)
        subject = m.group(1) if m else ""
        if not _looks_like_team(subject):
            return {"type": "player_goal_involvement", "player": _clean_player_name(subject)}
        # Subject is a team name -> fall through to team_score handler below.

    # "Will <player> score a goal (excluding own goals)?"  (not "any player")
    # This is a PURE goal market — distinct from "score or assist" above. It must
    # NOT include an assist term, so it routes to player_goal (goal only), not
    # player_goal_involvement.
    if "score a goal" in q and "excluding own goals" in q and "any player" not in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score a goal", q)
        subject = m.group(1) if m else ""
        if not _looks_like_team(subject):
            return {"type": "player_goal", "player": _clean_player_name(subject)}
        # Subject is a team name -> re-route as team_score below.

    # "Will any player score more than 1 goal (excluding own goals)?" (a brace)
    if "any player" in q and "score" in q and (
        "more than 1 goal" in q or "2 or more goals" in q or "brace" in q
    ):
        return {"type": "any_player_brace"}

    # "Will any player record 2 or more shots on target?" OR
    # "Will any <team> player have 2 or more shots on target?"
    if "any" in q and "player" in q and ("shot on target" in q or "shots on target" in q):
        n, direction = _threshold(q)
        return {"type": "any_player_sot", "threshold": n or 2, "direction": direction or "over"}

    # "Will <player> have at least 1 shot on target?" OR
    # "Will <player> (Country) have N or more shots on target?"
    # A parenthesized nation tag (or the "at least"/"a shot" phrasing) is the
    # signal this is a single player, not a whole team. Comparatives ("...than
    # ...") and "both teams"/halftime markets are handled elsewhere.
    if (("shot on target" in q or "shots on target" in q) and "have" in q
            and "than" not in q and "both teams" not in q and "halftime" not in q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+have", q)
        subject = m.group(1) if m else ""
        has_country = bool(re.search(r"\([^)]+\)", subject))
        # Route to player_shot_on_target only when we're confident the subject
        # is a person, not a team. See _looks_like_team for the heuristic.
        if has_country or (not _looks_like_team(subject) and ("at least" in q or "or more" in q or "have a shot" in q)):
            n, direction = _threshold(q)
            return {"type": "player_shot_on_target", "player": _clean_player_name(subject),
                    "threshold": n or 1, "direction": direction or "over", "half": _half(q)}

    # ---- Player vs player: more shots on target ----------------------------
    # "Will <player> record more shots on target than <player>?" Two individual
    # players (usually with (Country) tags), not teams.
    if "more shots on target than" in q and "record" in q:
        m = re.search(r"will\s+(.+?)\s+record more shots on target than\s+(.+?)\s*\??$", q)
        if m and not _looks_like_team(m.group(1)) and not _looks_like_team(m.group(2)):
            return {"type": "player_vs_player_sot",
                    "player": _clean_player_name(m.group(1)),
                    "opponent": _clean_player_name(m.group(2))}

    # ---- N+ distinct players from one team take a shot ---------------------
    # "Will 5 or more different <team> players attempt a shot?"
    if "different" in q and "player" in q and ("attempt a shot" in q or "take a shot" in q
                                               or "attempt a shot" in q or "have a shot" in q):
        n, direction = _threshold(q)
        return {"type": "distinct_shooters", "threshold": n or 5, "direction": direction or "over"}

    # ---- Goal in a specific time window ------------------------------------
    # "Will a goal be scored before the first hydration break / in stoppage time?"
    if "goal be scored" in q or ("goal" in q and "scored" in q):
        if "before the first hydration break" in q or "before first hydration break" in q:
            return {"type": "goal_before_hydration"}
        if "after the second hydration break" in q or "after second hydration break" in q:
            return {"type": "goal_after_hydration"}
        # "during first- or second-half stoppage time" — covers BOTH halves'
        # added time. Must be checked before the single-half branches, since
        # "second-half stoppage" is a substring of this combined phrase.
        if "stoppage" in q and (
            "first- or second" in q or "second- or first" in q
            or ("first half" in q and "second half" in q)
            or ("first-half stoppage" in q and "second-half stoppage" in q)
        ):
            return {"type": "goal_stoppage_time"}
        if "first-half stoppage" in q or "first half stoppage" in q:
            return {"type": "goal_first_half_stoppage"}
        if "second-half stoppage" in q or "second half stoppage" in q:
            return {"type": "goal_second_half_stoppage"}
        # "in the first half after the first hydration break" — window ~22'-45'
        if "first half" in q and "after the first hydration break" in q:
            return {"type": "goal_first_half_after_hydration"}
        # "after the first hydration break but before the second hydration break"
        # — window ~22' to ~67', the middle portion of the match
        if "after the first hydration break" in q and "before the second hydration break" in q:
            return {"type": "goal_between_breaks"}

    # ---- Half vs half goals comparison -------------------------------------
    # "Will the second half have/produce more goals than the first half?"
    # Must come BEFORE the generic "more X than" comparative.
    if "more goals than" in q and "first half" in q and "second half" in q:
        m = re.search(
            r"will the (first|second) half (?:have|produce) more goals than the (first|second) half", q
        )
        if m and m.group(1) != m.group(2):
            return {"type": "half_vs_half_goals", "more_half": m.group(1)}

    # ---- Comparative "more X than opponent" --------------------------------
    # "Will <team> have/commit/score more <metric> than <opp> [in the X half]?"
    comp = re.search(
        r"will\s+" + _TEAM_RE +
        r"\s+(?:have|commit|finish with|receive|get|score)\s+more\s+(shots on target|corner kicks|fouls|cards|goals)\s+than\s+" +
        _TEAM_RE + r"(?:\s+in the (?:first|second) half)?\??$",
        q,
    )
    if comp:
        return {
            "type": "team_more_than_opponent",
            "team": comp.group(1).title(),
            "metric": comp.group(2),
            "opponent": comp.group(3).title(),
            "half": _half(q),
        }

    # ---- Halftime markets --------------------------------------------------
    if "halftime" in q or "at halftime" in q:
        if "be tied" in q or "match be tied" in q:
            return {"type": "halftime_tied"}
        # "Will the match be 0-0 at halftime?" — explicit scoreline, same as tied
        if re.search(r"0.?0\s+at\s+halftime|halftime.*0.?0", q):
            return {"type": "halftime_tied"}
        if "both teams have at least" in q and "shot on target" in q:
            return {"type": "halftime_both_sot"}
        # "<team> be winning/ahead/leading" and bare-verb phrasings
        # ("<team> lead/leads/leading" or "<team> win/wins/winning") at halftime.
        if re.search(r"\b(?:be\s+)?(?:winning|ahead|leading|lead|leads|win|wins)\b", q):
            m = re.search(
                r"will\s+" + _TEAM_RE +
                r"\s+(?:be\s+)?(?:winning|ahead|leading|lead|leads|win|wins)\b", q)
            return {"type": "halftime_winning", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- Regulation ends in a draw (knockout) ------------------------------
    if ("end in a tie" in q or "end in a draw" in q) and "halftime" not in q:
        return {"type": "match_draw"}

    # ---- Match goes to extra time (knockout) -------------------------------
    # Equivalent to "regulation ends level" -> reuse the match_draw model.
    if "extra time" in q and ("go to extra time" in q or "reach extra time" in q
                              or "go to extra-time" in q):
        return {"type": "match_draw"}

    # ---- Total substitutions (knockout) ------------------------------------
    # "Will there be 9 or more total substitutions?" — near-certain given 5 subs
    # allowed per team; a flat base rate market.
    if "substitution" in q and ("total" in q or "combined" in q or "there be" in q):
        n, direction = _threshold(q)
        return {"type": "total_subs", "threshold": n or 9, "direction": direction or "over"}

    # ---- Goalkeeper saves (knockout) ---------------------------------------
    # "Will <keeper> make 4 or more saves?" — driven by the opponent's SOT.
    if "save" in q and "make" in q:
        n, direction = _threshold(q)
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+make", q)
        return {"type": "goalkeeper_saves", "player": _clean_player_name(m.group(1) if m else ""),
                "threshold": n or 3, "direction": direction or "over"}

    # ---- Both halves have the same number of goals (knockout) --------------
    if "both halves" in q and "same number of goals" in q:
        return {"type": "both_halves_same_goals"}

    # ---- Advance / qualify to the next round (knockout) --------------------
    if "advance to" in q or "advance past" in q or "qualify for" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+(?:advance|qualify)", q)
        return {"type": "team_advance", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- First goal is credited with an assist -----------------------------
    # "Will the first goal of the match be credited with an assist?"
    if "first goal" in q and "assist" in q and (
            "credited" in q or "with an assist" in q or "have an assist" in q):
        return {"type": "first_goal_assisted"}

    # ---- First goal --------------------------------------------------------
    # "Will <team> score the first goal of the (match/second half)?"
    if "score the first goal" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score the first goal", q)
        return {"type": "team_first_goal", "team": _title(m.group(1)) if m else "Unknown", "half": _half(q)}

    # ---- Substitute scores / substitution before halftime ------------------
    if "substitute" in q and "score" in q:
        return {"type": "sub_scores"}
    if "substitution" in q and ("before halftime" in q or "before half" in q or "in the first half" in q):
        return {"type": "sub_before_half"}

    # ---- Clean sheet -------------------------------------------------------
    if "clean sheet" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+keep a clean sheet", q)
        return {"type": "team_clean_sheet", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- Team scores in both halves ----------------------------------------
    if "score in both halves" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score in both halves", q)
        return {"type": "team_score_both_halves", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- Team scores N or more goals ---------------------------------------
    mg = re.search(r"will\s+" + _TEAM_RE + r"\s+score\s+(\d+)\s+or\s+more\s+goals", q)
    if mg:
        return {"type": "team_goals_over", "team": _title(mg.group(1)),
                "threshold": int(mg.group(2)), "direction": "over"}

    # ---- Team scores in a half ---------------------------------------------
    # "Will <team> score in the second half?"
    if re.search(r"score in the (?:first|second) half", q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score in the", q)
        return {"type": "team_score_half", "team": _title(m.group(1)) if m else "Unknown", "half": _half(q)}

    # ---- Team scores (full match) ------------------------------------------
    # "Will <team> score at least 1 goal?" / "score a goal"
    if re.search(r"score (?:at least \d+ goal|a goal)", q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score", q)
        return {"type": "team_score", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- Win by a margin (knockout) ----------------------------------------
    # "Will <team> win by 2 or more goals?"
    mb = re.search(r"will\s+" + _TEAM_RE + r"\s+win by\s+(\d+)\s+or\s+more\s+goals", q)
    if mb:
        return {"type": "team_win_by_margin", "team": _title(mb.group(1)), "margin": int(mb.group(2))}

    # ---- Win both halves (knockout) ----------------------------------------
    # "Will either team win both halves?" — a distinct market from match_winner.
    # Must come before the generic "win" branch below.
    if "win both halves" in q:
        return {"type": "win_both_halves"}

    # ---- Team holds a lead at any point (knockout) -------------------------
    # "Will Belgium hold a lead at any point?" is equivalent to "Will Belgium
    # score at least once?" — if they score, they were ahead at some point.
    # "excluding penalty shootout" is a qualifier, not a separate market.
    if "hold a lead" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+hold a lead", q)
        return {"type": "team_score", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- Substitution at halftime (knockout) --------------------------------
    if "substitution" in q and "halftime" in q:
        return {"type": "sub_at_halftime"}

    # ---- First goal in second half (knockout) ------------------------------
    # "Will the first goal be scored in the second half?"
    if "first goal" in q and "second half" in q and "other than" not in q:
        return {"type": "first_goal_second_half"}

    # ---- Match decided by exactly one goal (knockout) ----------------------
    # "Will the match be decided by exactly one goal?"
    if ("decided by" in q and "one goal" in q) or ("exactly one goal" in q and "decided" in q):
        return {"type": "match_decided_one_goal"}

    # ---- Card shown in each half (knockout) --------------------------------
    # "Will at least one card be shown in each half?"
    if "card" in q and "each half" in q:
        return {"type": "card_each_half"}

    # ---- Goal scored in each half (knockout) -------------------------------
    # "Will at least one goal be scored in each half?" — goals analogue of
    # card_each_half. P(≥1 goal 1H) × P(≥1 goal 2H).
    if "goal" in q and "each half" in q:
        return {"type": "goal_each_half"}

    # ---- Card shown in stoppage time (knockout) ----------------------------
    # "Will a card be shown during first- or second-half stoppage time?"
    if "card" in q and "stoppage time" in q and "total" not in q:
        return {"type": "card_stoppage_time"}

    # ---- Win the trophy (final): includes extra time + penalties -----------
    # "Will <team> win the World Cup / the final / the tournament / the title /
    # lift the trophy / be crowned champions?" A regulation draw is still
    # resolved via ET/penalties, so this is NOT the regulation-only match_winner.
    if ("win the world cup" in q or "win the final" in q or "win the tournament" in q
            or "win the title" in q or "lift the trophy" in q or "be crowned" in q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+(?:win|lift|be crowned)", q)
        return {"type": "match_winner_incl_et", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- Match winner ------------------------------------------------------
    # "Will <team> win the match?" / "Will <team> win [in regulation]?"
    if "win the match" in q or re.search(r"will\s+.+?\s+win\b", q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+win", q)
        return {"type": "match_winner", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- Each team records N+ shots on target (joint, full match) ----------
    # "Will each team record 4 or more shots on target?" — a joint market:
    # P(home >= N SOT) x P(away >= N SOT). The halftime version ("both teams ...
    # at halftime") is handled earlier in the halftime block.
    if ("shots on target" in q or "shot on target" in q) and (
            "each team" in q or "both teams" in q) and "than" not in q:
        n, direction = _threshold(q)
        return {"type": "both_teams_sot", "threshold": n or 1, "direction": direction or "over"}

    # ---- Team total shots on target (a named team) -------------------------
    # "Will <team> have 3 or more shots on target?" (players handled above; a
    # subject that isn't one of the two teams is rerouted to a player market in
    # submit.py).
    if ("shots on target" in q or "shot on target" in q) and "have" in q and "than" not in q:
        n, direction = _threshold(q)
        if n is not None:
            m = re.search(r"will\s+" + _TEAM_RE + r"\s+have", q)
            return {"type": "team_total_sot", "team": _title(m.group(1)) if m else "Unknown",
                    "threshold": n, "direction": direction, "half": _half(q)}

    # ---- Total shots, on and off target (match) ----------------------------
    if "total shots" in q and "on and off target" in q:
        n, direction = _threshold(q)
        if n is not None:
            return {"type": "total_shots", "threshold": n, "direction": direction, "half": _half(q)}

    # ---- Total shots on target (match) -------------------------------------
    if "total shots on target" in q:
        n, direction = _threshold(q)
        if n is not None:
            return {"type": "total_sot", "threshold": n, "direction": direction, "half": _half(q)}

    # ---- Total corners (both teams) ----------------------------------------
    # "Will there be 9 or more total corner kicks?"
    if "corner" in q and ("total corner" in q or "there be" in q):
        n, direction = _threshold(q)
        if n is not None:
            return {"type": "total_corners", "threshold": n, "direction": direction, "half": _half(q)}

    # ---- Team corners ------------------------------------------------------
    # "Will <team> have 5 or more corner kicks?" (singular "corner kick" too)
    if "corner" in q:
        n, direction = _threshold(q)
        if n is not None:
            m = re.search(r"will\s+" + _TEAM_RE + r"\s+have", q)
            return {"type": "team_corners", "team": _title(m.group(1)) if m else "Unknown",
                    "threshold": n, "direction": direction, "half": _half(q)}

    # ---- Total offsides (both teams) ---------------------------------------
    # "Will there be 3 or more offside calls?"
    if "offside" in q and "be caught offside" not in q and (
        "there be" in q or "offside calls" in q or "total offside" in q
    ):
        n, direction = _threshold(q)
        if n is not None:
            return {"type": "total_offsides", "threshold": n, "direction": direction, "half": _half(q)}

    # ---- Team offsides -----------------------------------------------------
    # "Will <team> be caught offside 2 or more times?"
    if "offside" in q:
        n, direction = _threshold(q)
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+be caught offside", q)
        return {"type": "team_offsides", "team": _title(m.group(1)) if m else "Unknown",
                "threshold": n or 2, "direction": direction or "over", "half": _half(q)}

    # ---- Both teams receive a card -----------------------------------------
    if "both teams" in q and "card" in q and "receive" in q:
        return {"type": "both_teams_card"}

    # ---- A red card shown --------------------------------------------------
    if "red card" in q:
        return {"type": "red_card"}

    # ---- A named team receives a card --------------------------------------
    # "Will <team> receive at least 1 card [in the second half]?"
    if "card" in q and re.search(r"will\s+" + _TEAM_RE + r"\s+receive", q) and "more" not in q:
        n, direction = _threshold(q)
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+receive", q)
        return {"type": "team_cards", "team": _title(m.group(1)) if m else "Unknown",
                "threshold": n or 1, "direction": direction or "over", "half": _half(q)}

    # ---- Total cards (match) ----------------------------------------------
    if "total cards" in q or "cards shown" in q:
        n, direction = _threshold(q)
        if n is not None:
            return {"type": "total_cards", "threshold": n, "direction": direction, "half": _half(q)}

    # ---- A card shown in the first half / late (after hydration break) -----
    if "card" in q and "hydration break" in q:
        return {"type": "card_late"}
    if "card" in q and "first half" in q:
        return {"type": "card_first_half"}

    # ---- Exact total goals (knockout) -------------------------------------
    # "Will exactly 1 goal be scored?"
    # "Will the match finish with exactly 2 total goals?"
    m_exact = re.search(r"exactly\s+(\d+)\s+(?:total\s+)?goals?\s+(?:be\s+scored|in\s+regulation)", q)
    if not m_exact:
        m_exact = re.search(r"finish with\s+exactly\s+(\d+)\s+(?:total\s+)?goals?", q)
    if m_exact:
        return {"type": "total_goals_exact", "n": int(m_exact.group(1))}

    # ---- Penalty shootout (knockout) --------------------------------------
    # "Will the match be decided by a penalty shootout?"
    # Guard: must be the main subject of the question, not a parenthetical
    # exclusion like "hold a lead at any point (excluding a penalty shootout)".
    if ("penalty shootout" in q or ("penalty" in q and "shootout" in q)):
        if "excluding" not in q and "decided by" in q:
            return {"type": "penalty_shootout"}

    # ---- VAR review (knockout) -----------------------------------------------
    # "Will the referee conduct an on-field review at the pitchside VAR monitor?"
    if "var" in q or "pitchside" in q or ("on-field review" in q and "referee" in q):
        return {"type": "var_review"}

    # ---- First substitution by a specific team (knockout) ------------------
    # "Will Spain make the first substitution of the match?"
    if "first substitution" in q and "make" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+make the first substitution", q)
        return {"type": "first_sub", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- First goal by single-digit shirt number (knockout) ----------------
    # "Will the first goal be scored by a player wearing a single-digit shirt number?"
    if "single-digit" in q and "shirt number" in q:
        return {"type": "first_goal_single_digit_shirt"}

    # ---- Total goals odd/even (knockout) -----------------------------------
    # "Will the total number of goals be an odd number?"
    if "total" in q and "goals" in q and "odd" in q:
        return {"type": "total_goals_odd"}
    if "total" in q and "goals" in q and "even" in q:
        return {"type": "total_goals_even"}

    # ---- First goal by a player other than named players -------------------
    # "Will the first goal be scored by a player other than X and Y?"
    if "first goal" in q and "other than" in q and "player" in q:
        return {"type": "first_goal_other_player"}

    # ---- Compound AND comparative (e.g. "more X AND more Y than opponent") --
    # "Will England have more corner kicks AND more total shots than Norway?"
    # Models as P(A wins metric1) × P(A wins metric2) with positive correlation
    # adjustment (both driven by same team dominance), applied before falling
    # through to unknown.
    if " and " in q and "more" in q and "than" in q and "corner" in q and "shot" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+have more", q)
        return {"type": "comparative_and", "team": _title(m.group(1)) if m else "Unknown"}

    # ---- Total goals over/under -------------------------------------------
    if "total goals" in q:
        n, direction = _threshold(q)
        half = _half(q)
        if n is not None:
            kind = "half_total_goals" if half else "total_goals"
            return {"type": kind, "threshold": n, "direction": direction, "half": half}

    # ---- "Will the <first/second> half produce N or more goals?" -----------
    mp = re.search(r"the (first|second) half produce", q)
    if mp:
        n, direction = _threshold(q)
        if n is not None:
            return {"type": "half_total_goals", "threshold": n, "direction": direction, "half": mp.group(1)}

    return {"type": "unknown", "raw": question}


if __name__ == "__main__":
    samples = [
        # group-stage style
        "Will Ghana win the match?",
        "Will the second half have 2 or more total goals?",
        "Will Antoine Semenyo have at least 1 shot on target?",
        "Will Panama be caught offside 2 or more times?",
        "Will Uzbekistan have 5 or more corner kicks?",
        # knockout style
        "Will Argentina win in regulation (90 minutes + stoppage time)?",
        "Will the United States win by 2 or more goals in regulation (90 minutes + stoppage time)?",
        "Will Japan advance to the Round of 16?",
        "Will both teams score in regulation (90 minutes + stoppage time)?",
        "Will Argentina keep a clean sheet in regulation (90 minutes + stoppage time)?",
        "Will Brazil score 2 or more goals in regulation (90 minutes + stoppage time)?",
        "Will Germany score in both halves in regulation (90 minutes + stoppage time)?",
        "Will regulation (90 minutes + stoppage time) end in a tie?",
        "Will France be ahead at halftime?",
        "Will Lionel Messi (Argentina) have 3 or more shots on target in regulation (90 minutes + stoppage time)?",
        "Will there be 9 or more total corner kicks in regulation (90 minutes + stoppage time)?",
        "Will there be 3 or more offside calls in regulation (90 minutes + stoppage time)?",
        "Will there be 22 or more total shots (on and off target) in regulation (90 minutes + stoppage time)?",
        "Will a goal be scored before the first hydration break?",
        "Will a red card be shown in the match?",
        "Will a substitution be made before halftime?",
        "Will any player record 2 or more shots on target in regulation (90 minutes + stoppage time)?",
        "Will the first half produce 2 or more goals?",
    ]
    for s in samples:
        print(f"{parse_question(s)['type']:24} | {s}")
