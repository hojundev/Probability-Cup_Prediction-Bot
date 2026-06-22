# Match-Bot

A second SportsPredict bot aimed at **per-match leaderboard prizes** (top 1 of a
single match's markets), not overall competition ranking. See
[`second_bot.md`](second_bot.md) for the full strategy.

It's a thin wrapper around `../competition-bot`: same model, same data, same
client. It adds two things:

1. **Selective extremizing** — pushes the competition-bot's probabilities
   further from 0.50 (logit stretch) on markets with genuine signal.
2. **A second API key** — runs as a separate bot under the same account.

## Layout

| File | Purpose |
|---|---|
| `extremize.py` | The `extremize(p, k)` logit-stretch transform. |
| `config.py` | Strategy knobs: `EXTREMIZE_K`, `PERIPHERAL_SHRINK`, `EXTREMIZE_TYPES`, `TARGET_MATCH`. |
| `match_bot.py` | Wires the competition-bot pipeline to the second key + extremizer. Single submission pass. |
| `scheduler.py` | Timed loop (full sweep every 2h). No lineup poll — keeps api-football quota use flat. |
| `tests/` | Unit + property-based tests for the extremizer and wrapper. |

## Setup

1. Install deps (shared with competition-bot):
   ```
   pip install -r requirements.txt
   ```
2. Add a **second** SportsPredict key to the project `.env` (distinct from
   `SPORTSPREDICT_KEY`):
   ```
   SPORTSPREDICT_KEY_BOT2=sp_live_...
   ```
   The bot refuses to run if this is missing or identical to the first key.

## Run

Single pass:
```
python match_bot.py
```

Continuous (scheduler — full sweep every 2h, no lineup poll):
```
python scheduler.py
```

## Tuning

- `EXTREMIZE_K` (default `1.4`) — higher = more aggressive / higher variance.
- `TARGET_MATCH` — set to a match name (e.g. `"Ghana vs Panama"`) to submit only
  for that match; `None` submits for all open matches.
- `EXTREMIZE_TYPES` — which market types get pushed. Player and penalty markets
  are intentionally excluded (noisy / signal-less).

Once per-match results accumulate, compare the match-bot's per-match RBP to the
competition-bot's on the same matches. If it consistently wins, raise `k`; if it
loses, lower it.

## Tests

```
python -m pytest tests/ -v
```
