import requests
import os
import json
import pandas as pd
from ift6758.features.ingenierie import features_live_game
from ift6758.data.cleaning import DataCleaner


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
        
        cleaned_new_plays = self.cleaner.extract_events(game_data, game_id, includeShootouts=False, keepPreviousEventInfo=False, includePowerPlay=False)
        
        self.game_plays_cache[game_id]['cleaned_plays'].extend(cleaned_new_plays)
        
        return most_recent_play_num

    def ping_game(self, game_id):
        if self.current_game_id != game_id:
            self.current_game_id = game_id
            if game_id not in self.game_plays_cache:
                self.game_plays_cache[game_id] = {
                    "cleaned_plays": [], 
                    "features": pd.DataFrame(),
                    "play_nums": [],
                }
        
        newest_play_num = self.update_new_game_plays(game_id)
        
        # fetch new plays and convert to DataFrame
        new_plays = pd.DataFrame(self.game_plays_cache[game_id]['cleaned_plays'][newest_play_num:])
        
        # clean new plays from bad data
        cleaned_new_plays = self.cleaner.remove_bad_data(new_plays)
        
        # update cache cleaned plays
        self.game_plays_cache[game_id]['cleaned_plays'][newest_play_num:] = cleaned_new_plays.to_dict('records')
        
        # generate features for the new plays
        new_plays_features = features_live_game(cleaned_new_plays)
        self.game_plays_cache[game_id]['features'] = pd.concat([self.game_plays_cache[game_id]['features'], new_plays_features])
        
        return self.game_plays_cache[game_id]


    def get_game_stats(self, game_id, predictions):
        
        team_names = (self.games_cached[game_id]['homeTeam']['name']['default'], self.games_cached[game_id]['awayTeam']['name']['default'])
        team_logos = (self.games_cached[game_id]['homeTeam']['logo'], self.games_cached[game_id]['awayTeam']['logo'])
        current_period = self.game_plays_cache[game_id]['cleaned_plays'][-1]['period']
        time_remaining = self.games_cached[game_id]['clock']['timeRemaining']
        score = (self.games_cached[game_id]['homeTeam']['score'], self.games_cached[game_id]['awayTeam']['score'])
        
        xg_home = []
        xg_away = []
        print(f"predictions: {predictions}")
        for i, play in enumerate(self.game_plays_cache[game_id]['cleaned_plays']):
            team = play.get('team')
            if team:
                if team == team_names[0]:
                    xg_home.append(predictions.iloc[i])
                else:
                    xg_away.append(predictions.iloc[i])
                    
        xg_home_total = round(sum(xg_home),2)
        xg_away_total = round(sum(xg_away),2)
        
        return {
            "team_names": team_names,
            "team_logos": team_logos,
            "current_period": current_period,
            "time_remaining": time_remaining,
            "score": score,
            "xG": [xg_home_total, xg_away_total]
        }