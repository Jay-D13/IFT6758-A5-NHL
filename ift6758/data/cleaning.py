import pandas as pd
import json
import pickle
import os
from ift6758.data import IGNORE_EVENTS, SeasonType
from ift6758.data.acquisition import NHLGameData

class DataCleaner:
    def __init__(self, data_raw: NHLGameData, data_path_clean):
        self.data_raw = data_raw # should be ..../json_raw/
        self.data_path_clean = data_path_clean # should be ..../json_clean/
        self.cache = {}
        
        os.makedirs(data_path_clean, exist_ok=True)
    
    def _get_from_cache(self, season: int) -> bool:
        """
            Retrieve clean data from cache for a specific season .
            Returns True if cache was found, else return False.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
        """

        cleaned_path = os.path.join(self.data_path_clean, str(season))
        file = os.path.join(cleaned_path, f"{season}.pkl")

        if os.path.exists(file):
            self.cache[season] = pd.read_pickle(file)
            return True
        return False

    def extract_events(self, data: dict, game_id :str) -> list[dict]:
        """
            Filters out events that are not shots or goals then extracts the relevant 
            information from the remaining events into a list of dictionaries.
            
            Args:
                game_path (str): The path to the game file.
                game_id (str): The game ID.
        """
            
        plays = data['liveData']['plays']['allPlays']
        events = []
        
        for event in plays:
            # Ignore events that are not shots or goals
            if event['result']['eventTypeId'] in IGNORE_EVENTS:
                continue
            
            try:
                data = {
                    'game_id': game_id,
                    'time': event['about']['periodTime'],
                    'event_type': event['result']['eventTypeId'],
                    'period': event['about']['period'],
                    'team': event['team']['name'],
                    'coordinates': event['coordinates'],
                    'shooter': next((p['player']['fullName'] for p in event['players'] if p['playerType'] in ['Shooter', 'Scorer']), None),
                    'goalie': next((p['player']['fullName'] for p in event['players'] if p['playerType'] == 'Goalie'), None),
                    'shot_type': event['result'].get('secondaryType', None),
                    'empty_net': event['result'].get('emptyNet', False), 
                    'strength': event['result'].get('strength', None), 
                }
                events.append(data)
            except KeyError as e:
                print(f"Failed to extract event data for game {game_id} due to missing key: {e}")
            except Exception as e:
                print(f"Failed to extract event data for game {game_id} due to unexpected error: {e}")
                
        return events
    
    def clean_season(self, season :int):
        """
            Extracts events from all games (playoffs and regular) in a given season and saves the data to a pickle file.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
        """ 
        if not self._get_from_cache(season):
            events = []
            for game_type in self.data_raw.data[season]:            
                for game_data in self.data_raw.data[season][game_type]:                
                    # Extract events from game and convert it to a DataFrame
                    game_id = game_data['gamePk']
                    events.extend(self.extract_events(game_data, game_id))
                    
            df = pd.DataFrame(events)
                    
            # Save to a pickle file
            self.save_cleaned_data(df, season)
            
            # Add to cache
            self.cache[season] = df
        
    def save_cleaned_data(self, df : pd.DataFrame, season: int):
        """
            Saves a DataFrame to a pickle file.
            
            Args:
                df (DataFrame): The DataFrame to save.
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                game_type (SeasonType): The type of game (regular or playoff).
                game_id (str): The game ID.
        """
        cleaned_path = os.path.join(self.data_path_clean, str(season))
        os.makedirs(cleaned_path, exist_ok=True)
        
        file = os.path.join(cleaned_path, f"{season}.pkl")
        df.to_pickle(file)