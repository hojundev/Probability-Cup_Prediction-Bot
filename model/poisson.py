import math
from scipy.stats import poisson


def predict_match_outcome(xg_home, xg_away, max_goals=10):
    """
    Predicts match outcome probabilities (home win, draw, away win) using Poisson distribution.
    """
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0
    
    for i in range(max_goals):
        for j in range(max_goals):
            prob_i_goals = poisson.pmf(i, xg_home)
            prob_j_goals = poisson.pmf(j, xg_away)
            joint_prob = prob_i_goals * prob_j_goals
            
            if i > j:
                home_win_prob += joint_prob
            elif i == j:
                draw_prob += joint_prob
            else:
                away_win_prob += joint_prob
                
    return {
        "home_win": home_win_prob,
        "draw": draw_prob,
        "away_win": away_win_prob
    }


def predict_btts(xg_home, xg_away):
    """
    Probability of Both Teams To Score (BTTS).
    """
    prob_home_no_score = poisson.pmf(0, xg_home)
    prob_away_no_score = poisson.pmf(0, xg_away)
    
    prob_home_score = 1 - prob_home_no_score
    prob_away_score = 1 - prob_away_no_score
    
    return prob_home_score * prob_away_score


def prob_at_least_one(mu):
    """P(count >= 1) for a Poisson(mu)."""
    return 1 - math.exp(-mu)


def prob_over_under(mu, threshold, direction, max_n=25):
    """
    Probability for an over/under count question on a single Poisson(mu).
      direction "over"  -> P(X >= threshold)
      direction "under" -> P(X <= threshold)
    """
    if direction == "under":
        return float(poisson.cdf(threshold, mu))
    # "over" means "threshold or more" => P(X >= threshold) = 1 - P(X <= threshold-1)
    return float(1 - poisson.cdf(threshold - 1, mu))


def prob_total_goals(xg_home, xg_away, threshold, direction):
    """Over/under on total match goals (sum of two independent Poissons)."""
    mu = xg_home + xg_away
    return prob_over_under(mu, threshold, direction)


def halftime_outcome_probs(xg_home, xg_away, first_half_share=0.42, max_goals=8):
    """
    Probabilities of the half-time state: (home_lead, tie, away_lead),
    using each team's first-half expected goals.
    """
    mu_h = xg_home * first_half_share
    mu_a = xg_away * first_half_share
    home_lead = tie = away_lead = 0.0
    for i in range(max_goals):
        pi = poisson.pmf(i, mu_h)
        for j in range(max_goals):
            pj = poisson.pmf(j, mu_a)
            joint = pi * pj
            if i > j:
                home_lead += joint
            elif i == j:
                tie += joint
            else:
                away_lead += joint
    return home_lead, tie, away_lead


def prob_x_greater_than_y(mu_x, mu_y, max_n=40):
    """
    P(X > Y) for two independent Poissons. Used for "Team A more <metric>
    than Team B" markets.
    """
    p = 0.0
    for x in range(max_n):
        px = poisson.pmf(x, mu_x)
        # P(Y <= x-1)
        if x >= 1:
            p += px * float(poisson.cdf(x - 1, mu_y))
    return p

if __name__ == "__main__":
    # Example: Brazil vs Serbia where Brazil has 2.1 xG and Serbia has 0.8 xG
    print("Match Outcome:", predict_match_outcome(2.1, 0.8))
    print("BTTS:", predict_btts(2.1, 0.8))
