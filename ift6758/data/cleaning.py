import pandas as pd
import os
from datetime import datetime
from ift6758.data import WANTED_EVENTS, PREVIOUS_EVENTS
from ift6758.data.acquisition import NHLGameData

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
        event_team_side = 'home' if event['team']['name'] == home_name else 'away'
        # Cas special pour les shootouts (periodType = SHOOTOUT) car period = 5 mais max(period) = 4
        period = periods_data[0] if event['about']['periodType'] == 'SHOOTOUT' else periods_data[event['about']['period']-1]
        
        home_rink_side = period['home'].get('rinkSide', None) if 'home' in period and 'rinkSide' in period['home'] else None
        away_rink_side = period['away'].get('rinkSide', None) if 'away' in period and 'rinkSide' in period['away'] else None
        
        # Check if rinkSide information is available
        if all([home_rink_side, away_rink_side]):
            return away_rink_side if event_team_side == 'home' else home_rink_side

        return None
    
    def _extract_event_data(self, event : dict, game_id : str, opposite_team_side : str) -> dict or None:
        try:
            shooter = next((p['player']['fullName'] for p in event['players'] if p['playerType'] in ['Shooter', 'Scorer']), None)
            goalie = next((p['player']['fullName'] for p in event['players'] if p['playerType'] == 'Goalie'), None)
            
            return {
                'game_id': game_id,
                'period': event['about']['period'],
                'period_time': event['about']['periodTime'],
                'event_type': event['result']['eventTypeId'],
                'team': event['team']['name'],
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
            return None
        except Exception as e:
            return None
        
    def _convert_time_to_seconds(self, time : str) -> int:
        minutes, seconds = map(int, time.split(':'))
        return minutes * 60 + seconds
        
    def _extract_previous_event_data(self, previous_event : dict, event : dict) -> dict or None:
        try:
            if 'coordinates' not in previous_event:
                x, y = None, None
            else:
                x = previous_event['coordinates'].get('x', None)
                y = previous_event['coordinates'].get('y', None)
            event_time = self._convert_time_to_seconds(event['period_time'])
            prev_time = previous_event['about']['periodTime']
            return {
                'prev_type': previous_event['result']['eventTypeId'],
                'prev_period_time': prev_time,
                'prev_x': x,
                'prev_y': y,
                'time_between_events': event_time - self._convert_time_to_seconds(prev_time),
                'distance_between_events': ((event['x'] - x)**2 + (event['y'] - y)**2)**0.5 if x is not None and y is not None else None,
            }
        except KeyError as e:
            return None
        except Exception as e:
            return None

    def extract_events(self, game_data: dict, game_id :str, includeShootouts : bool, keepPreviousEventInfo :bool) -> list[dict]:
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
        previous_event = None

        events = []
        for event in plays:           
            # Ignore shootouts
            if not includeShootouts and event['about']['periodType'] == 'SHOOTOUT':
                continue
        
            # Ignore events that are not shots or goals and update previous event
            if event['result']['eventTypeId'] not in WANTED_EVENTS:
                previous_event = event
                continue
            
            # Ignore events that do not have team side information (bad data) (it also coincidentally avoids overtime periods that don't end when going to shootouts)
            opposite_team_side = self._find_opposite_team_side(event, game_periods_info, home_name)
            if opposite_team_side is None: # It's always the whole game that is missing this information
                print(f"Failed to extract event data for game {game_id} due to missing rink side information")
                return None

            event_data = self._extract_event_data(event, game_id, opposite_team_side)
            if event_data is None:
                continue
            
            # Add previous event information
            if keepPreviousEventInfo and previous_event is not None:
                previous_event_data = self._extract_previous_event_data(previous_event, event_data)
                if previous_event_data is not None:
                    event_data.update(previous_event_data)
                
            events.append(event_data)
            previous_event = event
                    
        return events
    
    def clean_season(self, season :int, includeShootouts :bool = True, keepPreviousEventInfo :bool = False):
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
                    events = self.extract_events(game_data, game_id, includeShootouts, keepPreviousEventInfo)
                    if events:
                        season_events.extend(self.extract_events(game_data, game_id, includeShootouts, keepPreviousEventInfo))
                    
            df = pd.DataFrame(season_events)
            
            # Remove events with missing data
            df = self.remove_bad_data(df)
            
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
        
    def remove_bad_data(self, df : pd.DataFrame):
        """
            Removes events with missing data from a DataFrame.
            We keep events with missing goalie since he can be out of the net.
            We keep events with missing strength since it is not always available.
            We keep previous events with missing data since it is not always available.
            
            Args:
                df (DataFrame): The DataFrame to clean.
        """
        # remove bad coords
        df = df[df['x'].notna()]
        df = df[df['y'].notna()]
        
        # remove bad shot types
        df = df[df['shot_type'].notna()]
        
        return df
        