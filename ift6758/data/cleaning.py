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
                'period_time': self._convert_time_to_seconds(event['about']['periodTime']),
                'type': event['result']['eventTypeId'],
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
            x = previous_event['coordinates']['x']
            y = previous_event['coordinates']['y']
            event_time = event['period_time'] * event['period']
            prev_time = self._convert_time_to_seconds(previous_event['about']['periodTime']) * previous_event['about']['period']
            return {
                'prev_type': previous_event['result']['eventTypeId'],
                'prev_x': x,
                'prev_y': y,
                'time_since_prev': event_time - prev_time,
                'distance_from_prev': round(((event['x'] - x)**2 + (event['y'] - y)**2)**0.5,2) if x is not None and y is not None else None,
            }
        except KeyError as e:
            return None
        except Exception as e:
            # will happen because we have a few events with missing coordinates but we clean the dataframe later so all is good
            return None
        
    def extract_penalty_info(self, game_data: dict) -> list[dict]:
        power_play_indices = game_data['liveData']['plays']['penaltyPlays']
        all_plays = game_data['liveData']['plays']['allPlays']
        
        penalties = []
        for i in power_play_indices:            
            penalty_seconds = all_plays[i]['result']['penaltyMinutes'] * 60
            penalty_start_time = self._convert_time_to_seconds(all_plays[i]['about']['periodTime'])
            penalty_end_time = penalty_start_time + penalty_seconds
            penalty_period = all_plays[i]['about']['period']
                
            penalty_player_team = all_plays[i]['team']['name']
            penalty_severity = all_plays[i]['result']['penaltySeverity']
            
            end_time_next_period = penalty_end_time - 1200 if penalty_end_time > 1200 else None
            end_period = penalty_period + 1 if penalty_end_time > 1200 else penalty_period
            
            penalties.append({
                'start_time': penalty_start_time,
                'end_time': penalty_end_time,
                'end_time_next_period': end_time_next_period,
                'start_period': penalty_period,
                'end_period': end_period,
                'team_name': penalty_player_team,
                'severity': penalty_severity,
            })
        
        return penalties
    
    def _penalty_is_active(self, penalty : dict, period : int, time : int) -> bool:
        if penalty['start_period'] == period:
            return penalty['start_time'] <= time <= penalty['end_time']
        elif penalty['end_period'] == period:
            return 0 <= time <= penalty['end_time_next_period']
        return False
    
    def extract_power_play_info(self, penalties: list[dict], event: dict, home_team_name : str) -> dict:
        
        strength = event['result'].get('strength', {}).get('name', None)
        if strength == 'Even':
            return {
            'PPActive': False,
            'PPTimeElapsed': 0,
            'HomeSkaters': 5,
            'AwaySkaters': 5,
        }
                
        current_time = self._convert_time_to_seconds(event['about']['periodTime'])
        current_period = event['about']['period']
        event_type = event['result']['eventTypeId']

        # Update penalties end times if goal is scored
        if event_type == 'GOAL':
            team_who_scored = event['team']['name']
            for p in penalties:
                if self._penalty_is_active(p, current_period, current_time) and p['team_name'] != team_who_scored:
                    
                    if p['severity'] == 'Minor':
                        if ['start_period'] == current_period:
                            p['end_time'] = current_time
                        elif p['end_period'] == current_period:
                            p['end_time_next_period'] = current_time
                        
                    elif p['severity'] == 'Double Minor':
                        mid = p['start_time'] + 120
                        if p['start_time'] <= current_time < mid: # scored in the first half of the double minor
                            p['end_time'] = p['end_time'] - (mid - current_time)
                        else:
                            p['end_time'] = current_time #scored in the second half; end the double minor penalty

        
        home_skaters = 5
        away_skaters = 5
        power_play_active = False
        power_play_time_elapsed = 0
        
        # Determine number of skaters for each team
        active_penalties = []
        for p in penalties:
            if self._penalty_is_active(p, current_period, current_time):
                
                penalized_team = 'home' if p['team_name'] == home_team_name else 'away'
                
                if p['severity'] in ['Minor', 'Double Minor', 'Major']: # Misconducts are not counted as removing a skater
                    if penalized_team == 'home':
                        home_skaters -= 1
                    else:
                        away_skaters -= 1
                    
                active_penalties.append(p)

        # Determine if any penalties are still active
        if active_penalties:
            power_play_active = True
            earliest_start_time = min(p['start_time'] for p in active_penalties)
            power_play_time_elapsed = current_time - earliest_start_time

        power_play_info = {
            'PPActive': power_play_active,
            'PPTimeElapsed': power_play_time_elapsed,
            # "une équipe n’aura pas moins de 4 patineurs, y compris le gardien sur la glace à un moment donné"
            'HomeSkaters': 3 if home_skaters < 3 else home_skaters,
            'AwaySkaters': 3 if away_skaters < 3 else away_skaters,
        }

        return power_play_info
            

    def extract_events(self, game_data: dict, game_id :str, includeShootouts : bool, keepPreviousEventInfo :bool, includePowerPlay : bool) -> list[dict]:
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
        
        penalties = self.extract_penalty_info(game_data)
        
        events = []
        for event in plays:
                 
            # Ignore shootouts
            if not includeShootouts and event['about']['periodType'] == 'SHOOTOUT':
                continue
                
            # Ignore events that are not shots or goals
            if event['result']['eventTypeId'] in WANTED_EVENTS:
                     
                # Ignore events that do not have team side information (bad data) (it also coincidentally avoids overtime periods that don't end when going to shootouts)
                opposite_team_side = self._find_opposite_team_side(event, game_periods_info, home_name)
                if opposite_team_side is None: # It's always the whole game that is missing this information
                    # print(f"Failed to extract event data for game {game_id} due to missing rink side information")
                    return None

                # Extract event information
                event_data = self._extract_event_data(event, game_id, opposite_team_side)
                if event_data is None:
                    continue
                
                # Extract power play information
                if includePowerPlay:
                    power_play_info = self.extract_power_play_info(penalties, event, home_name)
                    # if power_play_info is not None:
                    event_data.update(power_play_info)
                            
                # Add previous event information
                if keepPreviousEventInfo and previous_event is not None:
                    previous_event_data = self._extract_previous_event_data(previous_event, event_data)
                    if previous_event_data is not None:
                        event_data.update(previous_event_data)
                    
                events.append(event_data)
            
            # update previous event
            if 'coordinates' in event:
                if len(event['coordinates']) == 2:
                    previous_event = event
        
        return events
    
    def clean_season(self, season :int, includeShootouts :bool = True, keepPreviousEventInfo :bool = False, includePowerPlay : bool = False):
        """
            Extracts events from all games (playoffs and regular) in a given season and saves the data to a pickle file.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
        """ 
        season_events = []
        for game_type in self.data_raw.data[season]:            
            for game_data in self.data_raw.data[season][game_type]:                
                # Extract events from game and convert it to a DataFrame
                game_id = game_data['gamePk']
                events = self.extract_events(game_data, game_id, includeShootouts, keepPreviousEventInfo, includePowerPlay)
                if events:
                    season_events.extend(self.extract_events(game_data, game_id, includeShootouts, keepPreviousEventInfo, includePowerPlay))
                
        df = pd.DataFrame(season_events)
        
        # Remove events with missing data
        df = self.remove_bad_data(df)
        
        # Save to a pickle file
        self.save_cleaned_data(df, season)
        
        # Add to cache
        self.cache[season] = df
            
    def get_cleaned_data(self, season: int) -> pd.DataFrame:
        if season not in self.cache:
            if not self._get_from_cache(season):
                self.clean_season(season)
                
        return self.cache[season]
    
    def get_team_games(self, season: int) -> dict[list]:
        """
            Returns a dictionary of the teams that played in a given season.
            The keys are the team names and the values are lists of game IDs.
        """
        output = {}
        
        rows = self.get_cleaned_data(season)
        team_games = rows.groupby('team')['game_id'].unique().to_dict()
        for team, games in team_games.items():
            regular = []
            playoffs = []
            for id in games:
                type = str(id)[4:6]
                if type == '02':
                    regular.append(id)
                elif type == '03':
                    playoffs.append(id)
                    
            assert len(regular) + len(playoffs) == len(games)
            
            if team not in output:
                output[team] = {'regular': [], 'playoffs': []}
            output[team] = {'regular': regular, 'playoffs': playoffs}
        
        return output
    
    def get_team_games_seasons(self, seasonStart: int, seasonEnd: int) -> dict[list]:
        """
            Returns a dictionary of the teams that played in a given season range.
            The keys are the team names and the values are lists of game IDs.
        """
        teams = {}
        for season in range(seasonStart, seasonEnd):
            team_games = self.get_team_games(season)
            for team, games in team_games.items():
                if team not in teams:
                    teams[team] = {'regular': [], 'playoffs': []}
                regular = games['regular']
                playoffs = games['playoffs']
                teams[team]['regular'].extend(regular)
                teams[team]['playoffs'].extend(playoffs)
            
        return teams
        
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
        
        # There's one previous event that's missing data
        df = df[df['prev_type'].notna()]
        
        return df
        
