import pandas as pd
import numpy as np
import os
from ift6758.data.acquisition import NHLGameData
from ift6758.data import NB_MAX_REGULAR_GAMES_PER_SEASON

def getdist(row)-> float:
    leftgoal_x = -90.0
    rightgoal_x = 90.0
    if row['opposite_team_side']=='left':
        return ((row['x']-leftgoal_x)**2+row['y']**2)**(1/2)
    if row['opposite_team_side']=='right':
        return ((row['x']-rightgoal_x)**2+row['y']**2)**(1/2)
    else:
        return None
        
    
class FeatureEng:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.cached_data = {}
        
    def fetch_data(self, startYear: int, endYear: int):
        
        if (startYear, endYear) in self.cached_data:
            return self.cached_data[(startYear, endYear)]
        
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
        
        trainValSets = self.fetch_data(startYear, endYear)
        
        columns_to_drop = ['game_id', 'time', 'period', 'team', 'coordinates', 'shooter', 'goalie', 'strength', 'shot_type']
        trainValSets.drop(columns=columns_to_drop, inplace=True)
        
        #1 or 0 for empty net
        trainValSets['empty_net'] = trainValSets['empty_net'].fillna(0)
        trainValSets['empty_net'] = trainValSets['empty_net'].astype(int)
        
        #1 or 0 for goal or shot respectively
        trainValSets['is_goal'] = trainValSets['event_type'].str.contains('GOAL').astype(int)
        trainValSets.drop(columns=['event_type'], inplace=True)
        trainValSets['distance'] = trainValSets.apply(getdist, axis = 1)
        trainValSets['angle'] = np.degrees(np.arcsin(trainValSets['y']/trainValSets['distance'])) # np.degrees(np.arctan2(train_val_sets['y'], train_val_sets['x'])) ?
        trainValSets.drop(columns = ['opposite_team_side', 'x', 'y'] , inplace=True)
        # train_val_sets.dropna(inplace=True) # Already managed in cleaning
        
        self.trainValSets = trainValSets
        self.trainValSets.to_pickle('./TrainValSets.pkl')
        return self.trainValSets
    
    def features_2(self, startYear: int, endYear: int):
        """
        À partir des données déjà triées, incluez les caractéristiques suivantes :
            ● Secondes de jeu (Game seconds)
            ● Période de jeu (Game period)
            ● Coordonnées (x,y, colonnes séparées)
            ● Distance de tir (Shot distance)
            ● Angle de tir (Shot angle)
            ● Type de tir (Shot type)
        """
        df = self.fetch_data(startYear, endYear)
        
        # secondes de jeu
    
    def getTestSet(self, year:int):
        file_path = os.path.join(self.data_path, str(year), f'{year}.pkl')
        if os.path.exists(file_path):
            self.testSet = pd.read_pickle(file_path)
            return self.testSet
        else:
            raise FileNotFoundError(f"No data found for year {year} at {file_path}.")
    