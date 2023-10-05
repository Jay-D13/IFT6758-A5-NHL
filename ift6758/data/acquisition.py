import os
import requests
import pickle
import json
import pandas as pd
from enum import Enum
from tqdm import tqdm
from ift6758.data import NB_MAX_REGULAR_GAMES_PER_SEASON, NB_MAX_PLAYOFF_GAMES_PER_SEASON, NHL_GAME_URL

class SeasonType(Enum):
    REGULAR = '02'
    PLAYOFF = '03'
    
class NHLGameData:
    def __init__(self, data_path, base_url=NHL_GAME_URL):
        self.base_url = base_url
        self.data_path = data_path
        self.cache = {}
        
        os.makedirs(data_path, exist_ok=True)
        
    def __add__(self, other):
        new_instance = NHLGameData(data_path=self.data_path)
        new_instance.cache = {**self.cache, **other.cache}
        return new_instance
    
    def _fetch_game_from_api(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve {url}. Status code: {response.status_code}")
        
        return response.json()
            
    def fetch_game(self, season, game_type, game_num):
        game_id = f"{season}{game_type.value}{game_num:04d}"
        url = self.base_url.format(GAME_ID=game_id)
        
        if game_id in self.cache:
            return self.cache[game_id]

        # Check locally
        local_file_path = os.path.join(self.data_path, f"{game_id}.json")
        if os.path.exists(local_file_path):
            with open(local_file_path, 'r') as file:
                data = json.load(file)
        else: # Get from API
            print(f"Retrieving data from: {url}...")
            data = self._fetch_game_from_api(url)
            
            with open(local_file_path, 'w') as out_file:
                json.dump(data, out_file)
                
        df = pd.DataFrame(data)
        self.cache[game_id] = df
        return df
    
    def fetch_season(self, season, regular_games=NB_MAX_REGULAR_GAMES_PER_SEASON, playoff_games=NB_MAX_PLAYOFF_GAMES_PER_SEASON):

        # Fetch regular season games
        for game_num in tqdm(range(1, regular_games + 1), desc="Fetching regular games"):
            print(f"Fetching game {game_num} of {regular_games}")
            self.fetch_game(season, SeasonType.REGULAR, game_num)
        
        # Fetch playoff games
        for game_num in tqdm(range(1, playoff_games + 1), desc="Fetching playoff games"):
            print(f"Fetching game {game_num} of {playoff_games}")
            self.fetch_game(season, SeasonType.PLAYOFF, game_num)
    