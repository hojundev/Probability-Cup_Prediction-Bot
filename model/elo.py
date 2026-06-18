class EloSystem:
    def __init__(self, k_factor=32):
        self.k_factor = k_factor
        self.ratings = {}

    def get_rating(self, team_name):
        # Default starting Elo is usually 1500
        return self.ratings.get(team_name, 1500)

    def set_rating(self, team_name, rating):
        self.ratings[team_name] = rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, team_a, team_b, actual_score_a):
        # actual_score_a: 1 for win, 0.5 for draw, 0 for loss
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        actual_score_b = 1 - actual_score_a
        
        new_rating_a = rating_a + self.k_factor * (actual_score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_score_b - expected_b)
        
        self.set_rating(team_a, new_rating_a)
        self.set_rating(team_b, new_rating_b)
        
        return new_rating_a, new_rating_b

# Example usage
if __name__ == "__main__":
    elo = EloSystem()
    elo.set_rating("Brazil", 1800)
    elo.set_rating("Serbia", 1600)
    print(f"Expected score Brazil vs Serbia: {elo.expected_score(1800, 1600)}")
