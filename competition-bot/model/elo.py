"""
Elo rating system for World Cup team strength, plus a helper that converts an
Elo matchup into an expected-goals (xG) split.

The ratings here are approximate World-Football-Elo style values for the 2026
World Cup field. They are a static prior: the bot blends a small slice of this
independent signal into the market-derived xG so that the model is not purely a
re-statement of the bookmaker line. Unknown teams default to 1500.
"""


# Approximate Elo ratings for the 2026 World Cup field. Keyed by the *normalized*
# country name (lowercase, accent-stripped, FIFA codes resolved) so they line up
# with bot.match_data.normalize_team_name. Values are a reasonable static prior;
# tune as results come in. Teams not listed fall back to DEFAULT_RATING.
WC2026_ELO = {
    "argentina": 2140,
    "france": 2050,
    "spain": 2045,
    "brazil": 2040,
    "england": 2000,
    "netherlands": 2030,
    "portugal": 1975,
    "belgium": 1930,
    "germany": 1960,
    "croatia": 1900,
    "uruguay": 1900,
    "colombia": 1915,
    "italy": 1925,
    "morocco": 1880,
    "switzerland": 1860,
    "denmark": 1850,
    "japan": 1850,
    "usa": 1800,
    "mexico": 1800,
    "senegal": 1820,
    "south korea": 1780,
    "ecuador": 1790,
    "austria": 1790,
    "sweden": 1770,
    "ukraine": 1770,
    "serbia": 1760,
    "poland": 1750,
    "peru": 1730,
    "wales": 1720,
    "scotland": 1710,
    "norway": 1760,
    "nigeria": 1740,
    "ivory coast": 1710,
    "egypt": 1700,
    "cameroon": 1690,
    "ghana": 1660,
    "australia": 1720,
    "canada": 1730,
    "iran": 1740,
    "saudi arabia": 1640,
    "qatar": 1600,
    "tunisia": 1660,
    "south africa": 1610,
    "panama": 1600,
    "costa rica": 1640,
    "paraguay": 1700,
    "venezuela": 1680,
    "jordan": 1560,
    "iraq": 1580,
    "uzbekistan": 1600,
    "new zealand": 1500,
    "haiti": 1490,
    "curacao": 1480,
    "cape verde": 1520,
    "jamaica": 1580,
    "honduras": 1560,
}

DEFAULT_RATING = 1500


class EloSystem:
    def __init__(self, k_factor=32):
        self.k_factor = k_factor
        self.ratings = {}

    def get_rating(self, team_name):
        # Default starting Elo is usually 1500
        return self.ratings.get(team_name, DEFAULT_RATING)

    def set_rating(self, team_name, rating):
        self.ratings[team_name] = rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def elo_xg_adjustment(self, home_rating, away_rating, base_total_xg):
        """
        Redistribute `base_total_xg` between the two teams according to their
        relative Elo strength. The stronger team gets the larger share.

        Returns (elo_xg_home, elo_xg_away). The total is conserved exactly:
        elo_xg_home + elo_xg_away == base_total_xg.
        """
        home_share = self.expected_score(home_rating, away_rating)
        elo_xg_home = base_total_xg * home_share
        elo_xg_away = base_total_xg - elo_xg_home  # exact conservation
        return elo_xg_home, elo_xg_away

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


def load_wc2026_ratings():
    """Return an EloSystem pre-populated with the 2026 World Cup field."""
    elo = EloSystem()
    for name, rating in WC2026_ELO.items():
        elo.set_rating(name, rating)
    return elo


# Example usage
if __name__ == "__main__":
    elo = load_wc2026_ratings()
    print("Brazil:", elo.get_rating("brazil"), "Serbia:", elo.get_rating("serbia"))
    print("Expected score Brazil vs Serbia:",
          elo.expected_score(elo.get_rating("brazil"), elo.get_rating("serbia")))
    print("xG split (total 2.6):",
          elo.elo_xg_adjustment(elo.get_rating("brazil"), elo.get_rating("serbia"), 2.6))
