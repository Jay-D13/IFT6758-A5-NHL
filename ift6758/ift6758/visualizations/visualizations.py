import pandas as pd
import os

def getdist(row)-> float:
    leftgoal_x = -90.0
    rightgoal_x = 90.0
    if row['opposite_team_side']=='left':
        return ((row['x']-leftgoal_x)**2+row['y']**2)**(1/2)
    if row['opposite_team_side']=='right':
        return ((row['x']-rightgoal_x)**2+row['y']**2)**(1/2)
    else:
        return None

class Visualizer:
    
    def __init__(self, data_path:str, season:int):
        self.df = pd.read_pickle(os.path.join(data_path, str(season), f'{season}.pkl'))
    
    def getShotProbabilities(self) -> pd.DataFrame:
        
        dfcop = self.df.copy()
        
        df_season_shots = dfcop.loc[:,['event_type', 'shot_type']]
                
        shotTypes = df_season_shots.groupby('shot_type').shot_type.agg('count')
        
        df_goals = df_season_shots.loc[df_season_shots['event_type'] == 'GOAL']
        
        goals = df_goals.groupby('shot_type').shot_type.agg('count')
    
        probabilities = (goals/shotTypes)
        
        finaldf = pd.concat([shotTypes,goals,probabilities],keys = ['shot_counts','goal_counts','probability_goal'], axis = 1)
        return finaldf


    def getShotDistances(self, decision: str = None) ->pd.DataFrame:         
        #now investigating the distances
        dfcop = self.df.copy()
        df_dist = dfcop.loc[:,['shot_type', 'event_type', 'x','y', 'opposite_team_side']]
        df_dist['distance_from_net'] = df_dist.apply(getdist, axis = 1)
        if(decision == 'GOAL'):
            df_goals = df_dist.loc[df_dist['event_type']=='GOAL']
            df_goals = df_goals.dropna()
            return df_goals
        if(decision == 'SHOT'):
            df_shots = df_dist.loc[df_dist['event_type']=='SHOT']
            df_shots = df_shots.dropna()
            return df_shots
        if(not decision):
            return df_dist.dropna()

    def getDistGoalProbabilities(self, bins: int)-> pd.DataFrame:
        dfcop = self.df.copy()
        df_dist = dfcop.loc[:,['shot_type', 'event_type', 'x','y', 'opposite_team_side']]
        df_dist['distance_from_net'] = df_dist.apply(getdist, axis = 1)
        df_dist = df_dist.dropna()
        #taking out the extreme cases
        df_dist = df_dist.loc[df_dist['distance_from_net']<=100.0]
        #df of all shot distances
        dfdist_all = df_dist.loc[:,['shot_type', 'event_type', 'distance_from_net']]
        
        #making bins for distances
        dfdist_bins = pd.cut(x = dfdist_all['distance_from_net'].to_numpy(), bins=bins, include_lowest = False, right = True)
        dfdist_all['dist_intervals'] = dfdist_bins
        
        #make a df with just the number of shots and their intervals
        hist_dists = dfdist_all.groupby([pd.Grouper(key='dist_intervals', sort = True), 'shot_type']).distance_from_net.count().reset_index(name = 'Total Shots')
        
        #make bins for goal distances
        df_goals = dfdist_all.loc[dfdist_all['event_type']=='GOAL']

        df_counts = hist_dists.groupby(['dist_intervals'])['Total Shots'].sum().reset_index(name = 'Total_Shots')
        arg = df_counts.Total_Shots.repeat(7).reset_index()

        #make df with the distance intervals and the number of goals
        hist_goaldists = df_goals.groupby([pd.Grouper(key = 'dist_intervals', sort = True),'shot_type']).distance_from_net.count().reset_index(name = 'Goals')
        
        hist_goaldists['Total Shots'] = arg['Total_Shots']
        hist_goaldists['Goals_over_Shots'] = hist_goaldists['Goals']/hist_goaldists['Total Shots']
        
        #returning a dataframe with the number of goals and total number of shots for each distance interval and the probability of that goal being that shot type

        return hist_goaldists




