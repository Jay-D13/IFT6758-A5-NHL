from enum import Enum

NHL_GAME_URL = 'https://api-web.nhle.com/v1/gamecenter/{GAME_ID}/play-by-play'
NB_MAX_REGULAR_GAMES_PER_SEASON = 1353
WANTED_EVENTS = ['shot-on-goal', 'goal', 'missed_shot']

class SeasonType(Enum):
    REGULAR = '02'
    PLAYOFF = '03'