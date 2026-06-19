"""
Standalone probability checker — reads ONLY from local cache, zero network calls.
Run with: python3 check_probs2.py
"""
import json, math, re, unicodedata
from scipy.stats import poisson

# ── Load cached odds ──────────────────────────────────────────────────────────
with open("data/.odds_cache.json") as f:
    raw = json.load(f)
odds_list = raw["data"]

# ── Minimal team-name normalizer (mirrors bot/match_data.py) ──────────────────
FIFA_CODES = {
    "CZE": "Czechia", "RSA": "South Africa", "SUI": "Switzerland",
    "BIH": "Bosnia and Herzegovina",
}
ALIASES = {
    "czechia": "czech republic", "turkiye": "turkey",
    "bosnia and herzegovina": "bosnia herzegovina",
    "united states": "usa", "korea republic": "south korea",
}

def norm(name):
    name = FIFA_CODES.get(name.strip().upper(), name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name)
    return ALIASES.get(name, name)

def split(match_name):
    parts = re.split(r"\s+vs\.?\s+|\s+v\s+|\s+-\s+", match_name, maxsplit=1, flags=re.I)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (match_name, "")

# ── Build odds index ──────────────────────────────────────────────────────────
index = {}
for ev in odds_list:
    home, away = ev.get("home_team"), ev.get("away_team")
    if not home or not away:
        continue
    h2h, totals = [], []
    for book in ev.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt["key"] == "h2h":
                by = {o["name"]: o["price"] for o in mkt["outcomes"]}
                prices = [by.get(home), by.get("Draw"), by.get(away)]
                if all(p and p > 1 for p in prices):
                    raw_p = [1/p for p in prices]
                    s = sum(raw_p)
                    h2h.append([r/s for r in raw_p])
            elif mkt["key"] == "totals":
                for o in mkt["outcomes"]:
                    if o.get("point"):
                        totals.append(float(o["point"]))
                        break
    if not h2h:
        continue
    n = len(h2h)
    ph = sum(r[0] for r in h2h)/n
    pd = sum(r[1] for r in h2h)/n
    pa = sum(r[2] for r in h2h)/n
    tg = sum(totals)/len(totals) if totals else None
    key = frozenset({norm(home), norm(away)})
    index[key] = dict(home=home, away=away, ph=ph, pd=pd, pa=pa, tg=tg)

def get_odds(match_name):
    h, a = split(match_name)
    return index.get(frozenset({norm(h), norm(a)}))

# ── Model helpers ─────────────────────────────────────────────────────────────
def xg(ph, pa, tg):
    total = tg or 2.6
    sup = ph - pa
    hs = max(0.15, min(0.85, 0.5*(1 + sup*0.55)))
    return max(0.2, total*hs), max(0.2, total*(1-hs))

def p1(mu): return 1 - math.exp(-mu)

def over_under(mu, n, d):
    return float(1 - poisson.cdf(n-1, mu)) if d=="over" else float(poisson.cdf(n, mu))

def total_goals(xh, xa, n, d): return over_under(xh+xa, n, d)

def btts(xh, xa): return p1(xh)*p1(xa)

def ht(xh, xa):
    mh, ma = xh*0.42, xa*0.42
    hl = ti = al = 0.0
    for i in range(8):
        for j in range(8):
            p = float(poisson.pmf(i,mh)) * float(poisson.pmf(j,ma))
            if i>j: hl+=p
            elif i==j: ti+=p
            else: al+=p
    return hl, ti, al

def fmt(p): return max(1, min(99, round(p*100)))

# ── Display ───────────────────────────────────────────────────────────────────
def show(match_name, ta, tb):
    o = get_odds(match_name)
    if not o:
        # Try flipped
        h,a = split(match_name)
        o = get_odds(f"{a} vs {h}")
    if not o:
        print(f"\nNo odds found for {match_name}")
        return
    xh, xa = xg(o['ph'], o['pa'], o['tg'])
    hl, ti, al = ht(xh, xa)

    rows = [
        (f"Will {ta} win the match?",                   fmt(o['ph'])),
        (f"Will {tb} win the match?",                   fmt(o['pa'])),
        ("Draw",                                         fmt(o['pd'])),
        (f"Will {ta} score at least 1 goal?",           fmt(p1(xh))),
        (f"Will {tb} score at least 1 goal?",           fmt(p1(xa))),
        ("Will both teams score (BTTS)?",               fmt(btts(xh, xa))),
        ("Will the match have 2 or more total goals?",  fmt(total_goals(xh,xa,2,'over'))),
        ("Will the match have 3 or more total goals?",  fmt(total_goals(xh,xa,3,'over'))),
        ("Will the match have 2 or fewer total goals?", fmt(over_under(xh+xa,2,'under'))),
        ("At halftime, will the match be tied?",        fmt(ti)),
        (f"At halftime, will {ta} be winning?",         fmt(hl)),
        (f"At halftime, will {tb} be winning?",         fmt(al)),
        (f"Will {ta} score in the second half?",        fmt(p1(xh*0.58))),
        (f"Will {tb} score in the second half?",        fmt(p1(xa*0.58))),
        ("Will a penalty OR red card be shown?",        40),
    ]

    print(f"\n{'='*65}")
    print(f"  {match_name}  ({ta} vs {tb})")
    print(f"  Odds: {ta}={o['ph']*100:.1f}%  Draw={o['pd']*100:.1f}%  {tb}={o['pa']*100:.1f}%")
    print(f"  Goals line: {o['tg']:.2f}  |  xG: {ta}={xh:.2f}  {tb}={xa:.2f}")
    print(f"{'='*65}")
    print(f"  {'Question':<55} {'Prob':>4}")
    print("  " + "-"*61)
    for q, p in rows:
        print(f"  {q:<55} {p:>4}")

show("CZE vs RSA", "Czechia", "South Africa")
show("SUI vs BIH", "Switzerland", "Bosnia-Herzegovina")
