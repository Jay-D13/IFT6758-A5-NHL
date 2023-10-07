import os
import requests
import pickle
import json
import pandas as pd
from tqdm import tqdm
from ift6758.data import NB_MAX_REGULAR_GAMES_PER_SEASON, NHL_GAME_URL, SeasonType
    
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

    def _ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)
    
    def _fetch_game_from_api(self, url : str):
        """
            Fetches game data from the NHL API and returns it as a dictionary.
            
            Args:
                url (str): The URL to fetch the data from.
        """
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve data from {url}.")
            return None

        return response.json()
    
    def fetch_game(self, season : int, game_type : SeasonType, game_num : str) -> dict:
        """
            Fetches game data from the NHL API and saves it to a local file.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                game_type (SeasonType): The type of game (regular or playoff).
                game_num (str): The game number (e.g. 0001 for the first game of the season).
        """
        game_id = f"{season}{game_type.value}{game_num}"
        url = self.base_url.format(GAME_ID=game_id)

        game_path = os.path.join(self.data_path, str(season), game_type.name.lower())
        self._ensure_dir(game_path)
        local_file_path = os.path.join(game_path, f"{game_id}.json")
        
        if os.path.exists(local_file_path):
            with open(local_file_path, 'r') as file:
                return json.load(file)
        else: 
            data = self._fetch_game_from_api(url)
            if data is None:
                return None
                
            with open(local_file_path, 'w') as out_file:
                json.dump(data, out_file)
        
        return data
                
    def fetch_playoff_games(self, season : int):
        """
            Fetches all playoff games for a given season.
            For playoff games, the 2nd digit of the specific number gives the round of the playoffs (1-4), 
            the 3rd digit specifies the matchup (out of 8), 
            and the 4th digit specifies the game (out of 7).
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
        """
        for round in range(1, 5):
            for matchup in range(1, 9):
                for game in range(1, 8):
                    if self.fetch_game(season, SeasonType.PLAYOFF, f"{round:02d}{matchup}{game}") is None:
                        print(f"No data in playoff game {round:02d}{matchup}{game} for season {season}.")
                        break

    def fetch_regular_games(self, season : int, num_games : int):
        """
            Fetches regular games for a given season.
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                num_games (int): The number of regular games to fetch.
        """
        for game_num in range(1, num_games + 1):
            if self.fetch_game(season, SeasonType.REGULAR, f"{game_num:04d}") is None:
                print(f"Failed to fetch regular game {game_num} for season {season}.")
                break
    
    def fetch_season(self, season : int, regular_games : int =NB_MAX_REGULAR_GAMES_PER_SEASON):
        """
            Fetches all games for a given season.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                regular_games (int): The number of regular games to fetch."""
        self.fetch_regular_games(season, regular_games)
        self.fetch_playoff_games(season)
    