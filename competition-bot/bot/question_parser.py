"""
Parses SportsPredict market questions into structured dicts the model can route.

The questions follow a fairly consistent template language. Each parse result
has a "type" plus the fields that type needs. Anything we can't confidently
classify returns {"type": "unknown"} so the caller can apply a neutral prior.

Ordering matters: more specific patterns are checked before generic ones.
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


def parse_question(question: str) -> dict:
    q = question.lower().strip()

    # ---- Compound questions we explicitly model ----------------------------
    # "Will both teams score AND the match have 3 or more total goals?"
    if "both teams score" in q and "total goals" in q:
        n, direction = _threshold(q)
        return {"type": "btts_and_total_goals", "threshold": n or 3, "direction": direction or "over"}

    # "Will a penalty kick be awarded OR a red card be shown ...?"
    if "penalty kick" in q and "red card" in q:
        return {"type": "penalty_or_red_card"}
    if "penalty kick be awarded" in q:
        return {"type": "penalty_awarded"}

    # Other AND/OR compounds we don't model precisely -> unknown (neutral prior)
    if (" and " in q or " or " in q) and "shot on target" not in q and "score or assist" not in q:
        # e.g. "Will Mexico score the first goal of the game and South Korea
        # score in the second half?" — too specific to model reliably.
        if "score the first goal" in q:
            return {"type": "unknown", "raw": question}

    # ---- Player markets ----------------------------------------------------
    # "Will <player> score or assist a goal (excluding own goals)?"
    if "score or assist" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score or assist", q)
        player = m.group(1).title() if m else "Unknown"
        return {"type": "player_goal_involvement", "player": player}

    # "Will <player> score a goal (excluding own goals)?"
    # The "(excluding own goals)" suffix is the reliable signal this is a
    # player market, not a team-scoring market. Must come before team_score.
    if "score a goal" in q and "excluding own goals" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score a goal", q)
        player = m.group(1).title() if m else "Unknown"
        return {"type": "player_goal_involvement", "player": player}

    # "Will <player> have at least 1 shot on target [in the second half]?"
    if "shot on target" in q and ("have at least" in q or "have a shot" in q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+have", q)
        player = m.group(1).title() if m else "Unknown"
        return {"type": "player_shot_on_target", "player": player, "half": _half(q)}

    # ---- Half vs half goals comparison -------------------------------------
    # "Will the second half have more goals than the first half?" (or reverse)
    # This must come BEFORE the generic "more X than" comparative, which would
    # otherwise misread the halves as two teams.
    if "more goals than" in q and "first half" in q and "second half" in q:
        m = re.search(
            r"will the (first|second) half have more goals than the (first|second) half", q
        )
        if m and m.group(1) != m.group(2):
            return {"type": "half_vs_half_goals", "more_half": m.group(1)}

    # ---- Comparative "more X than opponent" --------------------------------
    # "Will <team> have more shots on target than <opp> [in the second half]?"
    # "Will <team> commit more fouls than <opp>?"
    # "Will <opp> finish with more corner kicks than <team>?"
    # "Will <team> receive more cards than <opp>?"
    comp = re.search(
        r"will\s+" + _TEAM_RE +
        r"\s+(?:have|commit|finish with|receive|get)\s+more\s+(shots on target|corner kicks|fouls|cards|goals)\s+than\s+" +
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
    if "at halftime" in q or "halftime" in q:
        if "be tied" in q or "match be tied" in q:
            return {"type": "halftime_tied"}
        if "be winning" in q:
            m = re.search(r"will\s+" + _TEAM_RE + r"\s+be winning", q)
            team = m.group(1).title() if m else "Unknown"
            return {"type": "halftime_winning", "team": team}
        if "both teams have at least" in q and "shot on target" in q:
            return {"type": "halftime_both_sot"}

    # ---- First goal --------------------------------------------------------
    # "Will <team> score the first goal of the second half?"
    if "score the first goal" in q:
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score the first goal", q)
        team = m.group(1).title() if m else "Unknown"
        return {"type": "team_first_goal", "team": team, "half": _half(q)}

    # ---- Team scores -------------------------------------------------------
    # "Will <team> score in the second half?"
    if re.search(r"score in the (?:first|second) half", q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score in the", q)
        team = m.group(1).title() if m else "Unknown"
        return {"type": "team_score_half", "team": team, "half": _half(q)}

    # "Will <team> score at least 1 goal?" / "score a goal"
    if re.search(r"score (?:at least \d+ goal|a goal)", q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+score", q)
        team = m.group(1).title() if m else "Unknown"
        return {"type": "team_score", "team": team}

    # ---- Match winner ------------------------------------------------------
    if "win the match" in q or ("win" in q and "regulation" in q):
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+win", q)
        team = m.group(1).title() if m else "Unknown"
        return {"type": "match_winner", "team": team}

    # ---- Team total shots on target / corners / offsides -------------------
    # "Will <team> have 3 or more shots on target?"
    if "shots on target" in q and ("have" in q):
        n, direction = _threshold(q)
        if n is not None:
            m = re.search(r"will\s+" + _TEAM_RE + r"\s+have", q)
            team = m.group(1).title() if m else "Unknown"
            return {"type": "team_total_sot", "team": team, "threshold": n,
                    "direction": direction, "half": _half(q)}

    # "Will <team> have 5 or more corner kicks?"
    if "corner kicks" in q:
        n, direction = _threshold(q)
        if n is not None:
            m = re.search(r"will\s+" + _TEAM_RE + r"\s+have", q)
            team = m.group(1).title() if m else "Unknown"
            return {"type": "team_corners", "team": team, "threshold": n,
                    "direction": direction, "half": _half(q)}

    # "Will <team> be caught offside 2 or more times?"
    if "offside" in q:
        n, direction = _threshold(q)
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+be caught offside", q)
        team = m.group(1).title() if m else "Unknown"
        return {"type": "team_offsides", "team": team, "threshold": n or 2,
                "direction": direction or "over", "half": _half(q)}

    # "Will <team> receive at least 1 card in the second half?"
    if "card" in q and re.search(r"will\s+" + _TEAM_RE + r"\s+receive", q):
        n, direction = _threshold(q)
        m = re.search(r"will\s+" + _TEAM_RE + r"\s+receive", q)
        team = m.group(1).title() if m else "Unknown"
        return {"type": "team_cards", "team": team, "threshold": n or 1,
                "direction": direction or "over", "half": _half(q)}

    # ---- Total goals over/under -------------------------------------------
    if "total goals" in q:
        n, direction = _threshold(q)
        half = _half(q)
        if n is not None:
            kind = "half_total_goals" if half else "total_goals"
            return {"type": kind, "threshold": n, "direction": direction, "half": half}

    # ---- Total shots on target (match) ------------------------------------
    if "total shots on target" in q:
        n, direction = _threshold(q)
        if n is not None:
            return {"type": "total_sot", "threshold": n, "direction": direction, "half": _half(q)}

    # ---- Total cards (match) ----------------------------------------------
    if "total cards" in q or ("cards shown" in q):
        n, direction = _threshold(q)
        if n is not None:
            return {"type": "total_cards", "threshold": n, "direction": direction, "half": _half(q)}

    return {"type": "unknown", "raw": question}


if __name__ == "__main__":
    samples = [
        "Will Ghana win the match?",
        "Will the second half have 2 or more total goals?",
        "Will Ghana have 3 or more shots on target?",
        "Will Antoine Semenyo have at least 1 shot on target?",
        "Will Panama score the first goal of the second half?",
        "Will Panama be caught offside 2 or more times?",
        "Will Colombia have more shots on target than Uzbekistan in the second half?",
        "Will Uzbekistan score at least 1 goal?",
        "At halftime, will the match be tied?",
        "At halftime, will Canada be winning?",
        "Will the match have 2 or fewer total goals?",
        "Will Colombia score in the second half?",
        "Will Uzbekistan have 5 or more corner kicks?",
        "Will Eldor Shomurodov score or assist a goal (excluding own goals)?",
        "Will both teams score AND the match have 3 or more total goals?",
        "Will a penalty kick be awarded OR a red card be shown in the match?",
        "Will Ghana commit more fouls than Panama?",
        "Will there be 4 or more total cards shown?",
    ]
    for s in samples:
        print(f"{parse_question(s)['type']:24} | {s}")
