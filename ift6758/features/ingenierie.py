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

    def TrainValSets(self, startYear: int, endYear: int):
        dfs = []
        for year in range(startYear, endYear):
            df_2 = pd.read_pickle(os.path.join(self.data_path, str(year), f'{year}.pkl'))
            df_2.game_id = df_2.game_id.astype(str)
            yearstr = str(year)
            #taking only the regular season for each year
            df_2 = df_2.loc[df_2.game_id.str.startswith(yearstr+'02')]
            df_2.game_id = df_2.game_id.astype(int)
            dfs.append(df_2)
        
        trainValSets = pd.concat(dfs, ignore_index = True)
        self.unclean = trainValSets.copy()
        
        trainValSets = trainValSets.drop(columns = ['game_id', 'time', 'period', 'team', 'coordinates', 'shooter', 'goalie', 'strength', 'shot_type'])
        
        #1 or 0 for empty net
        trainValSets['empty_net'] = trainValSets['empty_net'].astype(int)
        trainValSets['empty_net'] = trainValSets['empty_net'].fillna(0)
        
        #1 or 0 for goal or shot respectively
        trainValSets['is_goal'] = trainValSets['event_type'].str.contains('GOAL').astype(int)
        trainValSets = trainValSets.drop(columns = ['event_type'])
        trainValSets['distance'] = trainValSets.apply(getdist, axis = 1)
        trainValSets['angle'] = np.degrees(np.arcsin(trainValSets.y/trainValSets.distance))
        trainValSets = trainValSets.drop(columns = ['opposite_team_side', 'x', 'y'])
        trainValSets = trainValSets.dropna()
        
        self.trainValSets  = trainValSets.copy()
        self.trainValSets.to_pickle('./TrainValSets.pkl')
        return self.trainValSets
    
    def getTestSet(self, year:int):
        self.TestSet = pd.read_pickle(os.path.join(self.data_path, str(year), f'{year}.pkl'))
        return self.TestSet
    