from enum import Enum

NHL_GAME_URL = 'https://statsapi.web.nhl.com/api/v1/game/{GAME_ID}/feed/live/'
NB_MAX_REGULAR_GAMES_PER_SEASON = 1353
WANTED_EVENTS = ['SHOT', 'GOAL']
PREVIOUS_EVENTS = ['SHOT', 'GOAL', 'MISSED_SHOT', 'BLOCKED_SHOT', 'FACEOFF', 'HIT', 'TAKEAWAY', 'GIVEAWAY']

class SeasonType(Enum):
    REGULAR = '02'
    PLAYOFF = '03'