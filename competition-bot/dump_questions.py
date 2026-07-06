"""
Diagnostic: dump every open market question and its parsed type.

Use this whenever a new round (e.g. Round of 32) goes live to find questions
the parser doesn't recognise yet. Unrecognised questions parse as "unknown" and
are currently submitted at a flat 0.50, so they need new parser rules + model
handlers before the bot is competitive on them.

Usage:
    python3 dump_questions.py            # all matches
    python3 dump_questions.py BRA        # only matches whose name contains BRA

Output:
    1. Per-match question list with parsed type.
    2. A count of questions by parsed type.
    3. A highlighted list of every DISTINCT question that parsed as "unknown".
"""

import sys
import logging
from collections import Counter, defaultdict

logging.basicConfig(level=logging.WARNING)  # suppress info noise

# Diagnostic tool — never write to the player cache.
import data.fetch_player_stats as _fps; _fps._api_disabled = True

from bot.client import get_probability_cup_lobby_and_event, fetch_matches, fetch_markets
from bot.question_parser import parse_question


def main():
    name_filter = " ".join(sys.argv[1:]).strip().lower()

    event, lobby = get_probability_cup_lobby_and_event()
    matches = fetch_matches(event["id"], lobby["id"])
    if name_filter:
        matches = [m for m in matches if name_filter in m.get("name", "").lower()]

    type_counts = Counter()
    unknown_questions = set()
    # Map every distinct question text -> parsed type, so we can eyeball new
    # phrasings even when they DID parse (in case they parsed to the wrong type).
    by_type = defaultdict(set)

    for match in sorted(matches, key=lambda x: x.get("name", "")):
        markets, _ = fetch_markets(lobby["id"], match["id"])
        print(f"\n=== {match.get('name', '?')} ===")
        for m in markets:
            q = m.get("question", "")
            qtype = parse_question(q).get("type", "unknown")
            type_counts[qtype] += 1
            by_type[qtype].add(q)
            if qtype == "unknown":
                unknown_questions.add(q)
            flag = "  <-- UNKNOWN" if qtype == "unknown" else ""
            print(f"  [{qtype:24}] {q}{flag}")

    print("\n" + "=" * 70)
    print("QUESTION TYPE COUNTS")
    print("=" * 70)
    for qtype, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {qtype:28} {n}")

    print("\n" + "=" * 70)
    print(f"UNRECOGNISED QUESTIONS ({len(unknown_questions)} distinct) "
          f"-- these submit at 0.50")
    print("=" * 70)
    if not unknown_questions:
        print("  (none — every question parsed to a known type)")
    else:
        for q in sorted(unknown_questions):
            print(f"  - {q}")


if __name__ == "__main__":
    main()
