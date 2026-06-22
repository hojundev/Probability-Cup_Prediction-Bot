def convert_odds_to_prob(decimal_odds):
    """
    Converts decimal odds to implied probability.
    """
    if decimal_odds <= 1.0:
        return 0.0 # Invalid odds
    return 1.0 / decimal_odds

def blend_probabilities(market_prob, model_prob, alpha=0.70):
    """
    Blends market probability and model probability.
    alpha: weight on market probability.
    """
    if market_prob is None:
        return model_prob
    
    return alpha * market_prob + (1 - alpha) * model_prob

def format_prediction_for_submission(prob):
    """
    Converts a decimal probability to an integer between 1 and 99.
    Edges (0 and 100) are not allowed by SportsPredict API.
    """
    int_prob = round(prob * 100)
    return max(1, min(99, int_prob))

if __name__ == "__main__":
    market = convert_odds_to_prob(2.5) # Implied 40%
    model = 0.50 # Implied 50%
    blended = blend_probabilities(market, model, alpha=0.7)
    final_sub = format_prediction_for_submission(blended)
    print(f"Market: {market:.2f}, Model: {model:.2f}, Blended: {blended:.2f}, Submission: {final_sub}")
