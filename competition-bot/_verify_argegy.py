import sys; sys.path.insert(0, '.')
from bot.question_parser import parse_question
from data.fetch_player_stats import peek_cache
from data.fetch_squads import resolve_player_team

questions = [
    "Will Argentina win in regulation (90 minutes + stoppage time)?",
    "Will Omar Marmoush (Egypt, #22) have 1 or more shots on target in regulation (90 minutes + stoppage time)?",
    "Will the total number of goals in regulation (90 minutes + stoppage time) be an odd number?",
    "Will Lionel Messi (Argentina, #10) score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?",
    "Will Mohamed Salah (Egypt, #10) score a goal (excluding own goals) in regulation (90 minutes + stoppage time)?",
    "Will there be 4 or more total cards shown in regulation (90 minutes + stoppage time)?",
    "Will both teams score in regulation (90 minutes + stoppage time)?",
    "Will Argentina have 5 or more shots on target in regulation (90 minutes + stoppage time)?",
    "Will Argentina have 6 or more corner kicks in regulation (90 minutes + stoppage time)?",
    "Will Julián Álvarez (Argentina, #9) score or assist a goal (excluding own goals) in regulation (90 minutes + stoppage time)?",
    "Will a goal be scored in the first half after the first hydration break?",
    "Will the match have 3 or more total goals in regulation (90 minutes + stoppage time)?",
    "Will the match be tied at halftime?",
    "Will the first goal of the match be scored by a player other than Lionel Messi and Mohamed Salah?",
    "Will any player score 2 or more goals (excluding own goals) in regulation (90 minutes + stoppage time)?",
]

print("=== Parser routing ===")
for q in questions:
    p = parse_question(q)
    t = p['type']
    subj = p.get('player', p.get('team', ''))
    flag = " <-- UNKNOWN" if t == 'unknown' else ""
    print(f"[{t:<26}] {subj:<25}{flag}")

print("\n=== Player cache + squad ===")
players = ['Omar Marmoush', 'Lionel Messi', 'Mohamed Salah', 'Julián Álvarez']
for name in players:
    cache = peek_cache(name)
    team = resolve_player_team(name)
    real = cache.get('is_real', False) if cache else 'NO CACHE'
    sot = cache.get('shots_on_target_per_90', '?') if cache else '?'
    shots = cache.get('shots_per_90', '?') if cache else '?'
    conv = cache.get('conversion_rate', '?') if cache else '?'
    xa = cache.get('xA_per_90', '?') if cache else '?'
    print(f"  {name:<22} team={str(team):<12} real={real}  sot={sot}  shots={shots}  conv={conv}  xA={xa}")
