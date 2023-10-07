from enum import Enum

NHL_GAME_URL = 'https://statsapi.web.nhl.com/api/v1/game/{GAME_ID}/feed/live/'
NB_MAX_REGULAR_GAMES_PER_SEASON = 1353
IGNORE_EVENTS = ['FAILED_SHOT_ATTEMPT', 'BLOCKED_SHOT', 'FACEOFF', 'STOP', 'SUB', 'FIGHT', 'GIVEAWAY', 'TAKEAWAY', 
                 'PERIOD_START', 'PERIOD_END', 'PERIOD_READY', 'PERIOD_OFFICIAL', 'GAME_END', 'GAME_OFFICIAL', 
                 'GAME_READY', 'GAME_SCHEDULED', 'EARLY_INT_START', 'EARLY_INT_END', 'SHOOTOUT_COMPLETE', 'CHALLENGE', 
                 'EMERGENCY_GOALTENDER']

class SeasonType(Enum):
    REGULAR = '02'
    PLAYOFF = '03'