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
        
        TrainValSets = pd.concat(dfs, ignore_index = True)
        self.unclean = TrainValSets.copy()
        
        TrainValSets = TrainValSets.drop(columns = ['game_id', 'time', 'period', 'team', 'coordinates', 'shooter', 'goalie', 'strength', 'shot_type'])
        
        #1 or 0 for empty net
        TrainValSets['empty_net'] = TrainValSets['empty_net'].astype(int)
        
        #1 or 0 for goal or shot respectively
        TrainValSets['is_goal'] = TrainValSets['event_type'].str.contains('GOAL').astype(int)
        TrainValSets = TrainValSets.drop(columns = ['event_type'])
        TrainValSets['distance'] = TrainValSets.apply(getdist, axis = 1)
        TrainValSets['angle'] = np.degrees(np.arcsin(TrainValSets.y/TrainValSets.distance))
        TrainValSets = TrainValSets.drop(columns = ['opposite_team_side', 'x', 'y'])

        self.TrainValSets  = TrainValSets.copy()
    
        return self.TrainValSets
    
    def getTestSet(self, year:int):
        self.TestSet = pd.read_pickle(os.path.join(self.data_path, str(year), f'{year}.pkl'))
        return self.TestSet
    