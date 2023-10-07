import pandas as pd
import json
import pickle
import os
from ift6758.data import IGNORE_EVENTS, SeasonType

class DataCleaner:
    def __init__(self, data_path_raw, data_path_clean):
        self.data_path_raw = data_path_raw # should be ..../json_raw/
        self.data_path_clean = data_path_clean # should be ..../json_clean/
        self.cache = {}
        
        os.makedirs(data_path_clean, exist_ok=True)
        
    def extract_events(self, game_path : str, game_id :str) -> list[dict]:
        """
            Filters out events that are not shots or goals then extracts the relevant 
            information from the remaining events into a list of dictionaries.
            
            Args:
                game_path (str): The path to the game file.
                game_id (str): The game ID.
        """
        with open(game_path) as f:
            data = json.load(f)
            
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
        season_path = os.path.join(self.data_path_raw, str(season))
        
        for game_type in SeasonType:
            game_type_path = os.path.join(season_path, game_type.name.lower())
            
            if not os.path.exists(game_type_path):
                print(f"No games found for {game_type.name.lower()} in season {season}.")
                continue
            
            for game_file in os.listdir(game_type_path):
                game_id, _ = os.path.splitext(game_file)
                game_path = os.path.join(game_type_path, game_file)
                
                # Extract events from game and convert it to a DataFrame
                events = self.extract_events(game_path, game_id)
                df = pd.DataFrame(events)
                
                # Save to a pickle file
                self.save_cleaned_data(df, season, game_type, game_id)
                
                # Add to cache
                self.cache[game_id] = df
        
    def save_cleaned_data(self, df : pd.DataFrame, season: int, game_type : SeasonType, game_id : str):
        """
            Saves a DataFrame to a pickle file.
            
            Args:
                df (DataFrame): The DataFrame to save.
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                game_type (SeasonType): The type of game (regular or playoff).
                game_id (str): The game ID.
        """
        cleaned_path = os.path.join(self.data_path_clean, str(season), game_type.name.lower())
        os.makedirs(cleaned_path, exist_ok=True)
        
        file = os.path.join(cleaned_path, f"{game_id}.pkl")
        with open(file, 'wb') as file:
            pickle.dump(df, file)