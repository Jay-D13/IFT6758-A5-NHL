import os, requests, tqdm
import pandas as pd
from ift6758.data import NB_MAX_REGULAR_GAMES_PER_SEASON, NB_MAX_PLAYOFF_GAMES_PER_SEASON

class NHLGameData:
    def __init__(self, data_path, base_url="https://statsapi.web.nhl.com/api/v1/game/{GAME_ID}/feed/live/"):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.base_url = base_url
        self.data_path = data_path
        self.cache = {}
        
    def __add__(self, other):
        new_instance = NHLGameData(data_path=self.data_path)
        new_instance.cache = {**self.cache, **other.cache}
        return new_instance
    
    def _fetch_game_from_api(self, url):
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve {url}. Status code: {response.status_code}")
        
        with open(self.data_path, 'w') as file:
            file.write(response.text)
        return response.json()
            
    
    def fetch_game(self, season, game_type, game_num):
        game_id = f"{season}0{game_type}0{game_num}"
        url = self.base_url.format(GAME_ID=game_id)
        
        if game_id in self.cache:
            return self.cache[game_id]
        
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as file:
                data = pd.read_json(file)
                self.cache[game_id] = data
                return data
        
        print(f"Retrieving data from '{url}'...")
        data = self._fetch_data_from_api(url)
        
        self.cache[game_id] = pd.DataFrame(data)
        return self.cache[game_id]
    
    def fetch_season(self, season, regular_games=NB_MAX_REGULAR_GAMES_PER_SEASON, playoff_games=NB_MAX_PLAYOFF_GAMES_PER_SEASON):

        # Fetch regular season games
        for game_num in tqdm(range(1, regular_games + 1)).set_description("Fetching regular season games"):
            print(f"Fetching game {game_num} of {regular_games}")
            self.fetch_game(season, 2, game_num)
        
        # Fetch playoff games
        for game_num in tqdm(range(1, playoff_games + 1)).set_description("Fetching playoff games"):
            print(f"Fetching game {game_num} of {playoff_games}")
            self.fetch_game(season, 3, game_num)
            