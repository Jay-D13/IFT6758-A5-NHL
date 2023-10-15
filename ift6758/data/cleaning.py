from datetime import datetime
import pandas as pd
import os
from ift6758.data import WANTED_EVENTS
from ift6758.data.acquisition import NHLGameData

DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
class DataCleaner:
    def __init__(self, data_raw: NHLGameData, data_path_clean):
        self.data_raw = data_raw
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

    def _find_opposite_team_side(self, event : dict, periods_data : dict, home_name : str) -> str or None:
        """
        Finds the opposite team side for a given event based on period data.

        Args:
            event (dict): The event data.
            periods_data (dict): Information about the periods.
            home_name (str): The name of the home team.
            
        Returns:
            str: The opposite team side or None.
        """
        event_datetime = datetime.strptime(event['about']['dateTime'], DATE_FORMAT)
        event_team_side = 'home' if event['team']['name'] == home_name else 'away'
        
        for period in periods_data:
            # Extract and check the validity of period information
            period_start = datetime.strptime(period['startTime'], DATE_FORMAT) if period.get('startTime') else None
            period_end = datetime.strptime(period['endTime'], DATE_FORMAT) if period.get('endTime') else None
            home_rink_side = period['home'].get('rinkSide', None) if 'home' in period and 'rinkSide' in period['home'] else None
            away_rink_side = period['away'].get('rinkSide', None) if 'away' in period and 'rinkSide' in period['away'] else None
            
            # Check if the event occurred during this period and if rinkSide information is available
            if all([period_start, period_end, home_rink_side, away_rink_side]) and period_start <= event_datetime <= period_end:
                return away_rink_side if event_team_side == 'home' else home_rink_side

        return None

    
    def _extract_event_data(self, event : dict, game_id : str, opposite_team_side : str) -> dict or None:
        """
        Extracts the relevant information from an event into a dictionary.
        
        Args:
            event (dict): The event data.
            game_id (str): The game ID.
            opposite_team_side (str): The opposite team side.
        
        Returns:
            dict: The extracted event data.
        """
        try:
            shooter = next((p['player']['fullName'] for p in event['players'] if p['playerType'] in ['Shooter', 'Scorer']), None)
            goalie = next((p['player']['fullName'] for p in event['players'] if p['playerType'] == 'Goalie'), None)
            
            return {
                'game_id': game_id,
                'time': event['about']['periodTime'],
                'event_type': event['result']['eventTypeId'],
                'period': event['about']['period'],
                'team': event['team']['name'],
                'coordinates': event['coordinates'],
                'x': event['coordinates'].get('x', None),
                'y': event['coordinates'].get('y', None),
                'shooter': shooter,
                'goalie': goalie,
                'shot_type': event['result'].get('secondaryType', None),
                'empty_net': event['result'].get('emptyNet', False),
                'strength': event['result'].get('strength', {}).get('name', None),
                'opposite_team_side': opposite_team_side,
            }
        except KeyError as e:
            print(f"Failed to extract event data for game {game_id} due to missing key: {e}")
            return None
        except Exception as e:
            print(f"Failed to extract event data for game {game_id} due to unexpected error: {e}")
            return None

    def extract_events(self, game_data: dict, game_id :str) -> list[dict]:
        """
            Filters out events that are not shots or goals then extracts the relevant 
            information from the remaining events into a list of dictionaries.
            
            Args:
                game_path (dict): The game data.
                game_id (str): The game ID.
        """
        plays = game_data['liveData']['plays']['allPlays']

        game_periods_info = game_data['liveData']['linescore']['periods']
        home_name = game_data['gameData']['teams']['home']['name']
        
        events = []
        for event in plays:
            # Ignore events that are not shots or goals
            if event['result']['eventTypeId'] not in WANTED_EVENTS:
                continue
            
            # Ignore events that do not have team side information (bad data)      
            opposite_team_side = self._find_opposite_team_side(event, game_periods_info, home_name)
            if opposite_team_side is None:
                print(f"Failed to extract event data for game {game_id} due to missing rink side information")
                return None

            event_data = self._extract_event_data(event, game_id, opposite_team_side)
            if event_data is not None:
                events.append(event_data)
                    
        return events
    
    def clean_season(self, season :int):
        """
            Extracts events from all games (playoffs and regular) in a given season and saves the data to a pickle file.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
        """ 
        if not self._get_from_cache(season):
            season_events = []
            for game_type in self.data_raw.data[season]:            
                for game_data in self.data_raw.data[season][game_type]:                
                    # Extract events from game and convert it to a DataFrame
                    game_id = game_data['gamePk']
                    events = self.extract_events(game_data, game_id)
                    if events:
                        season_events.extend(self.extract_events(game_data, game_id))
                    
            df = pd.DataFrame(season_events)
                    
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