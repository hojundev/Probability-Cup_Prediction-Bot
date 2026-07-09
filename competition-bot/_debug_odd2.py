import sys; sys.path.insert(0, '.')
from bot.question_parser import parse_question
from bot.submit import DEFAULT_PROB, _model_prob_for_market
from bot.match_data import build_odds_index
from data.fetch_odds import fetch_market_odds

# Step 1: confirm parser
q = "Will the total number of goals in regulation (90 minutes + stoppage time) be an odd number?"
parsed = parse_question(q)
print("1. parsed type:", parsed['type'])

# Step 2: check what _model_prob_for_market returns without live odds
fake_market = {"id": "test", "question": q, "match": {"name": "ARG vs EGY"}}
prob_no_odds = _model_prob_for_market(fake_market, {})
print("2. prob with empty odds_index:", prob_no_odds)

# Step 3: check if total_goals_odd is in PERIPHERAL_TYPES or any shrink list
from bot.submit import PERIPHERAL_TYPES, PERIPHERAL_SHRINK_OVERRIDES
print("3. in PERIPHERAL_TYPES:", 'total_goals_odd' in PERIPHERAL_TYPES)
print("3. in SHRINK_OVERRIDES:", 'total_goals_odd' in PERIPHERAL_SHRINK_OVERRIDES)

# Step 4: check MARKET_ALPHA blend path
# The 88/12 blend: if there's no matching odds line, does it return DEFAULT_PROB?
from bot.submit import MARKET_ALPHA
print("4. MARKET_ALPHA:", MARKET_ALPHA)
print("4. DEFAULT_PROB:", DEFAULT_PROB)
