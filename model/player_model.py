import math

def probability_player_scores(team_xg, player_shot_share, player_conversion_rate):
    """
    Probability of a player scoring at least one goal.
    """
    player_xg = team_xg * player_shot_share * player_conversion_rate
    return 1 - math.exp(-player_xg)

def probability_player_shots_on_target(team_xsot, player_sot_share):
    """
    Probability of a player getting at least one shot on target.
    """
    player_xsot = team_xsot * player_sot_share
    return 1 - math.exp(-player_xsot)

def probability_player_assist(team_xg, player_assist_rate):
    """
    Probability of a player getting an assist.
    """
    player_xa = team_xg * player_assist_rate
    return 1 - math.exp(-player_xa)

if __name__ == "__main__":
    # Example: Team xG = 2.0, Player takes 30% of shots, converts 15%
    prob = probability_player_scores(2.0, 0.3, 0.15)
    print(f"Probability Player Scores: {prob:.4f}")
