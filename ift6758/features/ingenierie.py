import pandas as pd
import numpy as np
import os
        
class FeatureEng:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.cached_data = {}
        
    def _get_dist_goal(self, side, x, y) -> float:
        goal_x = {
            'left': -90.0,
            'right': 90.0
        }
        
        if side not in goal_x:
            return None
        
        return round(((x - goal_x[side]) ** 2 + y ** 2) ** 0.5,2)
        
    def _fetch_data(self, startYear: int, endYear: int) -> pd.DataFrame:
        
        if (startYear, endYear) in self.cached_data:
            return self.cached_data[(startYear, endYear)].copy()
        
        dfs = []
        for year in range(startYear, endYear):
            file_path = os.path.join(self.data_path, str(year), f'{year}.pkl')
            if os.path.exists(file_path):
                df = pd.read_pickle(file_path)
                df['game_id'] = df['game_id'].astype(str)
                # taking only the regular season for each year
                df = df[df['game_id'].str.startswith(f'{year}02')]
                df['game_id'] = df['game_id'].astype(int)
                dfs.append(df)
            else:
                print(f"File not found: {file_path}")
        
        data = pd.concat(dfs, ignore_index = True)
        self.cached_data[(startYear, endYear)] = data
        
        return data.copy()

    def features_1(self, startYear: int, endYear: int):
        
        trainValSets = self._fetch_data(startYear, endYear)
        
        #1 or 0 for empty net
        trainValSets['empty_net'] = trainValSets['empty_net'].fillna(0)
        trainValSets['empty_net'] = trainValSets['empty_net'].astype(int)
        
        #1 or 0 for goal or shot respectively
        trainValSets['is_goal'] = trainValSets['type'].str.contains('GOAL').astype(int)
        trainValSets.drop(columns=['type'], inplace=True)
        trainValSets['distance_goal'] = trainValSets.apply(lambda row: self._get_dist_goal(row['opposite_team_side'], row['x'], row['y']), axis=1)
        trainValSets['angle_shot'] = np.where(trainValSets['distance_goal'] == 0, 
                                         0, 
                                         np.degrees(np.arcsin(trainValSets['y'] / trainValSets['distance_goal'])))
        
        anomalies = trainValSets.loc[trainValSets['distance_goal']>=100]
        anomalies = anomalies.loc[anomalies['is_goal']==1]
        anomalies = anomalies.loc[anomalies['empty_net']==0]
        anorm_columns_to_drop = ['period_time', 'game_time', 'period', 
                           'team', 'shooter', 'goalie', 'strength', 'shot_type',
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
        df_distGoal_counts = df_dist_goals.groupby(['distance_range'])['is_goal'].count().reset_index(name = 'Total_Goals_Bin')
        df_distGoal_rate = pd.DataFrame()
        df_distGoal_rate['distances'] = df_distGoal_counts['distance_range']
        df_distGoal_rate['GoalDist_Rate'] = df_distGoal_counts['Total_Goals_Bin']/shot_counts
        
        #df_angle_counts = df_angles.groupby(['angle_range'])['is_goal'].count().reset_index(name='Total_Shots_Bin')
        df_angleGoal_counts = df_angle_goals.groupby(['angle_range'])['is_goal'].count().reset_index(name='Total_Goals_Bin')
        df_angleGoal_rate = pd.DataFrame()
        df_angleGoal_rate['angles'] = df_angleGoal_counts['angle_range']
        df_angleGoal_rate['GoalAngle_Rate'] = df_angleGoal_counts['Total_Goals_Bin']/shot_counts
        
        final_df = pd.concat([df_distGoal_rate, df_angleGoal_rate], axis=1)
        
        return final_df

    def features_2(self, startYear: int, endYear: int):
        df = self._fetch_data(startYear, endYear)
        
        # Convert period_time of event to total game time and rename to 'game_seconds'
        df['period_time'] = df['period_time'].astype(str)
        df['period_time'] = df['period_time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
        df['period_time'] = df['period_time'] + (df['period'] * 20 * 60)
        df.rename(columns={'period_time': 'game_seconds'}, inplace=True)
        
        # Distance
        df['distance_goal'] = df.apply(lambda row: self._get_dist_goal(row['opposite_team_side'], row['x'], row['y']), axis=1)
        df['prev_distance_goal'] = df.apply(lambda row: self._get_dist_goal(row['opposite_team_side'], row['prev_x'], row['prev_y']), axis=1)
        
        # Angle
        df['angle_shot'] = np.where(df['distance_goal'] == 0, 0, round(np.degrees(np.arcsin(df['y'] / df['distance_goal'])),2))
        df['prev_angle_shot'] = np.where(df['prev_distance_goal'] == 0, 0, round(np.degrees(np.arcsin(df['prev_y'] / df['prev_distance_goal'])),2))
    
        # Rebond
        event_types = ['SHOT', 'MISSED_SHOT', 'BLOCKED_SHOT']
        df['bounce'] = df['prev_type'].isin(event_types)
        
        # Changement d'angle de tir
        df['angle_change'] = np.where((df['bounce']), df['angle_shot'] - df['prev_angle_shot'], 0)    
            
        # Vitesse
        df['speed'] = np.where(df['time_since_prev'] > 0, 
                           round(df['distance_from_prev'] / df['time_since_prev'],2), 
                           'instant')
        
        columns_to_drop = ['type', 'team', 'shooter', 'goalie', 'strength', 
                           'empty_net', 'opposite_team_side', 'prev_period_time']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        df.reset_index(drop=True, inplace=True)
    
        return df
    
    def getTestSet(self, year:int):
        file_path = os.path.join(self.data_path, str(year), f'{year}.pkl')
        if os.path.exists(file_path):
            self.testSet = pd.read_pickle(file_path)
            return self.testSet
        else:
            raise FileNotFoundError(f"No data found for year {year} at {file_path}.")
    