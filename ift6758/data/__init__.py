"""
You can "pre" import stuff in the __init__.py module, but this is not required.
This allows you to import this function as:

    from ift6758.data import get_player_stats

instead of only:

    from ift6758.data.question_1 import get_player_stats

but both are still valid. You can read more about packages and modules here:
https://docs.python.org/3/reference/import.html#regular-packages
"""
from .question_1 import get_player_stats

NHL_GAME_URL = 'https://statsapi.web.nhl.com/api/v1/game/{GAME_ID}/feed/live/'
NB_MAX_REGULAR_GAMES_PER_SEASON = 1353
NB_MAX_PLAYOFF_GAMES_PER_SEASON = 130
