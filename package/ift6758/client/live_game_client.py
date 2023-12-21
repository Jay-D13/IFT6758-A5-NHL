import requests
import os
import json
import pandas as pd
import logging
from ift6758.features.ingenierie import features_live_game
from ift6758.data.cleaning import DataCleaner

logger = logging.getLogger(__name__)

class LiveGameClient:
    def __init__(self):
        self.base_url = "https://api-web.nhle.com/v1/gamecenter"
        self.cleaner = DataCleaner(data_raw=None, data_path_clean="./") # don't need args for live game
        self.current_game_id = None
        self.game_plays_cache = {}
        self.games_cached = {}

    def update_new_game_plays(self, game_id):
        url = f"{self.base_url}/{game_id}/play-by-play"
        most_recent_play_num = self.game_plays_cache[game_id]['play_nums'][-1] if self.game_plays_cache[game_id]['play_nums'] else 0

        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            self.games_cached[game_id] = data
            game_data = {
                'plays': data['plays'][most_recent_play_num:],
                'homeTeam': data['homeTeam'],
                'awayTeam': data['awayTeam'],
                'rosterSpots': data['rosterSpots'],
            }

        for i, play in enumerate(game_data['plays']):
            if play['typeDescKey'] not in ['goal', 'shot-on-goal', 'missed_shot']:
                continue
            self.game_plays_cache[game_id]['play_nums'].append(most_recent_play_num + i + 1)
        
        logger.info("Empty plays" if not game_data['plays'] else f"New plays: {len(game_data['plays'])}")
        cleaned_new_plays = self.cleaner.extract_events(game_data, game_id, includeShootouts=False, keepPreviousEventInfo=False, includePowerPlay=False)
        logger.info(f"New plays: {cleaned_new_plays}")
        
        self.game_plays_cache[game_id]['cleaned_plays'].append(cleaned_new_plays)
        
        return most_recent_play_num

    def get_prediction(self, features):
        ip = os.environ.get('SERVING_IP')
        port = os.environ.get('SERVING_PORT')
        url = f"http://{ip}:{port}/predict"
        response = requests.post(
            url,
            json=json.loads(features.to_json())
        )
        
        if response.status_code == 200:
            return response.json()['predictions'][0][1]
        else:
            return None

    def ping_game(self, game_id):
        if self.current_game_id != game_id:
            self.current_game_id = game_id
            if game_id not in self.game_plays_cache:
                self.game_plays_cache[game_id] = {
                    "cleaned_plays": [], 
                    "features": pd.DataFrame(),
                    "play_nums": [],
                    "predictions": []
                    }
                
        newest_play_num = self.update_new_game_plays(game_id)
        game_plays = pd.DataFrame(self.game_plays_cache[game_id]['cleaned_plays'][newest_play_num:])
        
        new_plays_features = features_live_game(game_plays)
        self.game_plays_cache[game_id]['features'] = pd.concat([self.game_plays_cache[game_id]['features'], new_plays_features])

        prediction = self.get_prediction(new_plays_features)
        self.game_plays_cache[game_id]['predictions'].append(prediction)
        
        return self.game_plays_cache[game_id], self.update_game_stats(game_id)

    def get_game_stats(self, game_id):
        
        team_names = (self.games_cached[game_id]['homeTeam']['name']['default'], self.games_cached[game_id]['awayTeam']['name']['default'])
        team_logos = (self.games_cached[game_id]['homeTeam']['logo'], self.games_cached[game_id]['awayTeam']['logo'])
        current_period = self.game_plays_cache[game_id]['cleaned_plays'][-1]['period']
        time_remaining = self.games_cached[game_id]['clock']['timeRemaining']
        score = (self.games_cached[game_id]['homeTeam']['score'], self.games_cached[game_id]['awayTeam']['score'])
        
        xg_home = []
        xg_away = []
        
        for i, play in enumerate(self.game_plays_cache[game_id]['cleaned_plays']):
            if play['team'] == team_names[0]:
                xg_home.append(self.game_plays_cache[game_id]['predictions'][i])
            else:
                xg_away.append(self.game_plays_cache[game_id]['predictions'][i])
                
        xg_home = sum(xg_home)
        xg_away = sum(xg_away)
        
        return {
            "team_names": team_names,
            "team_logos": team_logos,
            "current_period": current_period,
            "time_remaining": time_remaining,
            "score": score,
            "xG": [xg_home, xg_away]
        }