# Jump Trading Probability Cup Bot

An automated forecasting bot for the [Jump Trading Probability Cup 2026](https://sportspredict.com) — a free-to-play forecasting contest during the 2026 FIFA World Cup (June 11 – July 19, 2026).

The bot ingests live betting odds, player stats, and confirmed lineups, runs a blended statistical model, and submits calibrated probability predictions (1–99) across ~485 binary yes/no markets on the SportsPredict platform. Scored by **Relative Brier Points** — the goal is calibration, not just picking winners.

---

## How It Works

```
Betting odds (The Odds API)
Player stats (api-football)        →  Poisson/Dixon-Coles model
Elo ratings (WC 2026 priors)       →  Blend (88% market / 12% model)
Confirmed lineups (api-football)   →  Shrinkage on peripheral markets
                                   →  Lineup-triggered re-scoring
SportsPredict markets              →  Question parser
                                   →  Calibrated integer (1–99)
                                   →  PATCH/POST via SportsPredict API
```

Each full run:
1. Discovers the Probability Cup event, joins the lobby
2. Fetches all 49 matches and their open markets (~485 total)
3. Fetches live betting odds (cached 2h to preserve quota)
4. Routes each market question to the right model output
5. Blends team xG with Elo-derived strength split
6. Applies shrinkage toward 50 for signal-less peripheral markets
7. PATCHes existing predictions / POSTs new ones
8. All requests paced by a central token bucket (≤55 req/min) — 429s auto-retry

A separate 15-minute lineup poll:
- Resolves each upcoming match to an api-football fixture
- Detects when confirmed starting XIs drop (90-min pre-kickoff window)
- Re-runs the model for that match; slashes benched/absent player markets to 30%
- Stops polling a match once its lineup is confirmed

---

## Setup

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env and fill in the 4 keys (see below)

# 3. Verify
python3 tests/test_model.py
```

### Required API Keys

| Key | Where to get it | Free tier |
|-----|----------------|-----------|
| `SPORTSPREDICT_KEY` | SportsPredict app → Profile → My Bots → Generate New Bot | 60 req/min |
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com) | 500 req/month |
| `API_FOOTBALL_KEY` | [dashboard.api-football.com](https://dashboard.api-football.com/register) | 100 req/day |
| `FOOTBALL_DATA_KEY` | [football-data.org](https://www.football-data.org/client/register) | 10 req/min |

---

## Running

```bash
# One-shot run (submits/updates all open predictions)
python3 -u -m bot.submit

# Continuous scheduler (full sweep every 2h, lineup poll every 15min)
python3 scheduler.py

# Look up questions + probabilities for a specific match
python3 lookup.py GER CIV
python3 lookup.py BRA Haiti
python3 lookup.py TUR PAR

# List all match names (to find the right codes)
python3 lookup.py
```

A full run takes ~10–15 minutes: fetching markets + PATCHing ~485 predictions at the rate-limited pace. 429s are automatically retried — the run is self-healing and will never crash from throttling.

---

## Project Structure

```
competition-bot/
├── .env                      # API keys (gitignored)
├── .env.example              # Template — copy this and fill in keys
├── requirements.txt
├── scheduler.py              # APScheduler — full sweep every 2h, lineup poll every 15min
├── lookup.py                 # CLI tool: python3 lookup.py TEAM1 TEAM2
├── bot/
│   ├── client.py             # SportsPredict API client (TokenBucket + retry)
│   ├── rate_limiter.py       # Central TokenBucket rate limiter
│   ├── question_parser.py    # Parses market questions into structured types
│   ├── match_data.py         # Odds indexing, team-name resolution, xG estimation
│   └── submit.py             # Main loop: fetch → model → PATCH/POST + lineup updates
├── data/
│   ├── fetch_matches.py      # football-data.org fixtures
│   ├── fetch_odds.py         # The Odds API (2-hour disk cache)
│   ├── fetch_player_stats.py # api-football player stats (permanent disk cache)
│   ├── fetch_lineups.py      # api-football confirmed starting XIs
│   ├── fetch_squads.py       # api-football squad lists
│   ├── .odds_cache.json      # Cached odds (gitignored)
│   └── .player_cache.json    # Cached player stats (gitignored)
├── model/
│   ├── poisson.py            # Dixon-Coles Poisson model + exact BTTS-and-over
│   ├── player_model.py       # Player goal/shot/assist probabilities
│   ├── elo.py                # Elo ratings — WC 2026 priors, wired into xG split
│   └── ensemble.py           # Blends market odds + model, formats to 1–99
└── tests/
    ├── test_model.py         # Core model unit tests
    ├── test_rate_limiter.py  # TokenBucket tests
    └── test_uzb_col.py       # Fixture-specific regression test
```

---

## Match Name Reference

The API uses FIFA 3-letter codes for most teams, but a few use full names:

| Full name | API name |
|-----------|----------|
| Haiti | `Haiti` |
| Curacao | `Curacao` |
| New Zealand | `New Zealand` |
| All others | 3-letter FIFA code (`BRA`, `GER`, `FRA`, etc.) |

---

## Model Details

### Scoring
`RBP = (crowd_brier − your_brier) × 100` with stage multipliers (group 1×, knockout 2×, final 3×). Calibration beats overconfidence.

### Question types covered (~97% of markets)
`match_winner`, `team_score`, `team_score_half`, `team_first_goal`, `total_goals`, `half_total_goals`, `btts_and_total_goals`, `halftime_tied`, `halftime_winning`, `halftime_both_sot`, `player_shot_on_target`, `player_goal_involvement`, `team_total_sot`, `total_sot`, `team_corners`, `team_offsides`, `team_cards`, `total_cards`, `team_more_than_opponent`, `penalty_awarded`, `penalty_or_red_card`

### Pipeline per market
1. Parse question text → structured type + parameters
2. Look up betting odds for the match (FIFA code resolution: `GHA` → `Ghana`)
3. Estimate total match xG from win probabilities + totals line
4. Split xG by team using Elo ratings (total conserved exactly)
5. Route to Poisson model / player stats / base-rate prior
6. Blend: `0.88 × market + 0.12 × model`
7. Apply shrinkage toward 50 on peripheral markets (corners, cards, offsides, "more than opponent")
8. Clamp to integer 1–99

### Key tunable constants (`bot/submit.py`)

| Constant | Value | Effect |
|----------|-------|--------|
| `MARKET_ALPHA` | `0.88` | Weight on sharp betting-market line vs model |
| `PERIPHERAL_SHRINK` | `0.35` | Fraction of deviation from 50 kept on peripheral markets (lower = closer to 50) |
| `FIRST_HALF_GOAL_SHARE` | `0.42` | Share of goals expected in the first half |
| `SOT_PER_XG` | `3.0` | Expected shots on target per expected goal |
| `PENALTY_AWARDED_RATE` | `0.26` | P(≥1 penalty awarded in a match) |
| `ELO_BLEND_WEIGHT` | `0.15` | Weight on Elo-derived xG split vs market-derived xG |
| `MAX_PLAYER_REQUESTS_PER_RUN` | `20` | Cap on live api-football player lookups per run |
| `BENCH_PLAYER_FACTOR` | `0.30` | Multiplier applied to player market prob when benched/absent |
| `LINEUP_WINDOW_MINUTES` | `90` | Pre-kickoff window during which lineup polling is active |
| `LINEUP_CHECK_INTERVAL_MINUTES` | `15` | Cadence of the lineup poll |

---

## API Quota Management

| API | Budget | Strategy |
|-----|--------|----------|
| SportsPredict | 60 req/min rolling | Central token bucket (55/min) + unlimited 429 retry |
| The Odds API | 500 req/month (resets July 1) | 2-hour disk cache (`data/.odds_cache.json`) |
| api-football | 100 req/day, 10 req/min | Permanent disk cache per player; miss caching; circuit breaker; 20 live lookups/run cap |
| football-data.org | 10 req/min | Rate-limit header inspection + 429 back-off |

---

## Known Good Facts

- All unit tests pass
- 50/50 matches resolve to betting odds (FIFA code + alias resolution)
- ~97% of market questions parse to a real type (not "unknown")
- Predictions span deciles 10–70 (properly calibrated, not all 50)
- 485 predictions successfully PATCHed in a complete run (June 17, 2026)
- 429s auto-retry — run is self-healing end to end
- Lineup polling live in the scheduler (15-min cadence, 90-min pre-kickoff window)
