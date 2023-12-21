import pandas as pd
import numpy as np
import os
        
class FeatureEng:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.cached_data = {}
        
    def _fetch_data(self, startYear: int, endYear: int, keepPlayoffs=False) -> pd.DataFrame:
        
        if (startYear, endYear, keepPlayoffs) in self.cached_data:
            return self.cached_data[(startYear, endYear, keepPlayoffs)].copy()
        
        dfs = []
        for year in range(startYear, endYear):
            file_path = os.path.join(self.data_path, str(year), f'{year}.pkl')
            if os.path.exists(file_path):
                df = pd.read_pickle(file_path)
                df['game_id'] = df['game_id'].astype(str)
                # taking only the regular season for each year
                if not keepPlayoffs:
                    df = df[df['game_id'].str.startswith(f'{year}02')]
                df['game_id'] = df['game_id'].astype(int)
                dfs.append(df)
            else:
                print(f"File not found: {file_path}")
        
        data = pd.concat(dfs, ignore_index = True)
        self.cached_data[(startYear, endYear, keepPlayoffs)] = data
        
        return data.copy()

    def features_1(self, startYear: int, endYear: int):
        
        trainValSets = self._fetch_data(startYear, endYear)
        
        #1 or 0 for empty net
        trainValSets['empty_net'] = trainValSets['empty_net'].fillna(0)
        trainValSets['empty_net'] = trainValSets['empty_net'].astype(int)
        
        #1 or 0 for goal or shot respectively
        trainValSets['is_goal'] = trainValSets['type'].str.contains('GOAL').astype(int)
        trainValSets.drop(columns=['type'], inplace=True)
        trainValSets['distance_goal'] = trainValSets.apply(lambda row: get_dist_goal(row['opposite_team_side'], row['x'], row['y']), axis=1)
        trainValSets['angle_shot'] = np.where(trainValSets['distance_goal'] == 0, 
                                         0, 
                                         np.degrees(np.arcsin(trainValSets['y'] / trainValSets['distance_goal'])))
        
        anomalies = trainValSets.loc[trainValSets['distance_goal']>=100]
        anomalies = anomalies.loc[anomalies['is_goal']==1]
        anomalies = anomalies.loc[anomalies['empty_net']==0]
        anorm_columns_to_drop = ['game_time', 'team', 'shooter', 'goalie', 'strength', 'shot_type',
                           'prev_type', 'prev_x', 'prev_y', 'time_since_prev', 'distance_from_prev',
                           'opposite_team_side', 'x', 'y', 'prev_period_time']
        anomalies.drop(columns=anorm_columns_to_drop, inplace=True, errors='ignore')
        anomalies = anomalies.sort_values(by=['distance_goal'])
        self.anomalies = anomalies

        columns_to_drop = ['game_id', 'period_time', 'game_time', 'period', 
                           'team', 'shooter', 'goalie', 'strength', 'shot_type',
                           'prev_type', 'prev_x', 'prev_y', 'time_since_prev', 'distance_from_prev',
                           'opposite_team_side', 'x', 'y', 'prev_period_time']
        
        
        trainValSets.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        
        self.trainValSets = trainValSets
        self.trainValSets.to_pickle('./TrainValSets.pkl')
        return self.trainValSets
    
    def getProbabilities(self, bins: int):
        
        df = self.trainValSets.copy()
        df_dist_bins = pd.cut(x=df['distance_goal'].to_numpy(), bins=bins, include_lowest = True, right = False)
        df_angle_bins = pd.cut(x=df['angle_shot'].to_numpy(), bins=bins, include_lowest = True, right = False)
        df['distance_range'] = df_dist_bins
        df['angle_range'] = df_angle_bins

        df_dist = df.loc[:,['is_goal', 'distance_range']]
        df_angles = df.loc[:,['is_goal', 'angle_range']]
        
        df_dist_goals = df_dist[df_dist['is_goal']==1]
        df_angle_goals = df_angles[df_angles['is_goal']==1]

        shot_counts = df['is_goal'].count()
        
        #df_dist_counts = df_dist.groupby(['distance_range'])['is_goal'].count().reset_index(name = 'Total_Shots_Bin')        
        df_distGoal_goalCounts = df_dist_goals.groupby(['distance_range'])['is_goal'].count().reset_index(name = 'Count_Goals_Bin')
        df_distGoal_totalCounts = df.groupby(['distance_range'])['is_goal'].count().reset_index(name = 'Total_Count_Goals_Bin')
        df_distGoal_rate = pd.DataFrame()
        df_distGoal_rate['distances'] = df_distGoal_goalCounts['distance_range']
        df_distGoal_rate['GoalDist_Rate'] = df_distGoal_goalCounts['Count_Goals_Bin'] / df_distGoal_totalCounts['Total_Count_Goals_Bin']
        
        #df_angle_counts = df_angles.groupby(['angle_range'])['is_goal'].count().reset_index(name='Total_Shots_Bin')
        df_angleGoal_goalCounts = df_angle_goals.groupby(['angle_range'])['is_goal'].count().reset_index(name='Count_Goals_Bin')
        df_angleGoal_totalCounts = df.groupby(['angle_range'])['is_goal'].count().reset_index(name='Total_Count_Goals_Bin')
        df_angleGoal_rate = pd.DataFrame()
        df_angleGoal_rate['angles'] = df_angleGoal_goalCounts['angle_range']
        df_angleGoal_rate['GoalAngle_Rate'] = df_angleGoal_goalCounts['Count_Goals_Bin'] / df_angleGoal_totalCounts['Total_Count_Goals_Bin']
        
        final_df = pd.concat([df_distGoal_rate, df_angleGoal_rate], axis=1)
        
        return final_df

    def features_2(self, startYear: int, endYear: int, drop_teams = True, keepPlayoffs=False):
        df = self._fetch_data(startYear, endYear, keepPlayoffs)
        
        # Convert period_time of event to total game time and rename to 'game_seconds'
        df['period_time'] = df['period_time'] + (df['period'] * 20 * 60)
        df.rename(columns={'period_time': 'game_seconds'}, inplace=True)
        
        # Distance
        df['distance_goal'] = df.apply(lambda row: get_dist_goal(row['opposite_team_side'], row['x'], row['y']), axis=1)
        df['prev_distance_goal'] = df.apply(lambda row: get_dist_goal(row['opposite_team_side'], row['prev_x'], row['prev_y']), axis=1)
        
        # Angle
        df['angle_shot'] = np.where(df['distance_goal'] == 0, 0, round(np.degrees(np.arcsin(df['y'] / df['distance_goal'])),2))
        df['prev_angle_shot'] = np.where(df['prev_distance_goal'] == 0, 0, round(np.degrees(np.arcsin(df['prev_y'] / df['prev_distance_goal'])),2))
    
        # Rebond
        event_types = ['SHOT', 'MISSED_SHOT', 'BLOCKED_SHOT']
        df['bounce'] = df['prev_type'].isin(event_types)
        
        # Changement d'angle de tir
        df['angle_change'] = np.where((df['bounce']), df['angle_shot'] - df['prev_angle_shot'], 0)    
            
        # Vitesse
        # Replace 0 with 1 to avoid division by 0
        df['time_since_prev'] = df['time_since_prev'].replace(0, 1)
        df['speed'] = round(df['distance_from_prev'] / df['time_since_prev'],2)
        
        # Add target column
        df['is_goal'] = df['type'].str.contains('GOAL').astype(int)
        
        columns_to_drop = ['strength','shooter', 'goalie', 'opposite_team_side', 'prev_period_time', 'type']
        if drop_teams:
            columns_to_drop += ['team']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')   
        
        df.reset_index(drop=True, inplace=True)
        return df
    
    def get_team_games(self, season: int) -> dict[list]:
        """
            Returns a dictionary of the teams that played in a given season.
            The keys are the team names and the values are lists of game IDs.
        """
        output = {}
        
        df = self._fetch_data(season, season+1, keepPlayoffs=True)
        team_games = df.groupby('team')['game_id'].unique().to_dict()
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
                    teams[team] = {}
                
                teams[team][season] = games
                            
        return teams
    
    def remove_first_team_games(self, df: pd.DataFrame, team_season_games: dict, num_regular : int = 5 , num_playoffs : int = 0):
        """
            Removes the first num_regular regular season games and num_playoffs playoff games for each team in the given dataframe.
        """
        for team, seasons in team_season_games.items():
            team_games_to_remove = []
            for season, games in seasons.items():
                team_games_to_remove += games['regular'][:num_regular]
                team_games_to_remove += games['playoffs'][:num_playoffs]
            
            df.drop(df[(df['team'] == team) & (df['game_id'].isin(team_games_to_remove))].index, inplace=True)
            
    def getTestSet(self, year:int):
        file_path = os.path.join(self.data_path, str(year), f'{year}.pkl')
        if os.path.exists(file_path):
            self.testSet = pd.read_pickle(file_path)
            return self.features_2(year, year+1, keepPlayoffs=True)
        else:
            raise FileNotFoundError(f"No data found for year {year} at {file_path}.")
    
    def encodeCategories(self, df: pd.DataFrame, categorical_features: list):
        return pd.get_dummies(df, columns=categorical_features, drop_first=True)
    

def features_live_game(game_events : pd.DataFrame): # new api (annoying) so we limit ourselves to the features we need for simple models   
     
    df = game_events.copy()
    print(df.columns, flush=True)
    df['distance_goal'] = df.apply(lambda row: get_dist_goal(row['opposite_team_side'], row['x'], row['y']), axis=1)
    df['angle_shot'] = np.where(df['distance_goal'] == 0, 0, round(np.degrees(np.arcsin(df['y'] / df['distance_goal'])),2))
    
    df['empty_net'] = df['empty_net'].fillna(0)
    df['empty_net'] = df['empty_net'].astype(int)
    
    return df[['distance_goal', 'angle_shot', 'empty_net']]

def get_dist_goal(side, x, y) -> float:
    goal_x = {
        'left': -90.0,
        'right': 90.0
    }
    
    if side not in goal_x:
        return None
    
    return round(((x - goal_x[side]) ** 2 + y ** 2) ** 0.5,2)