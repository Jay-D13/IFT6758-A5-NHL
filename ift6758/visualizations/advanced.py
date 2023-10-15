import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import numpy as np

class AdvancedVisualization:
    def __init__(self, data_path:str):
        self.data_path = data_path
    
    def load_season_data(self, season:int) -> pd.DataFrame:
        path = self.data_path.format(season=season)
        return pd.read_pickle(path)
    
    def adjust_coordinates(self, df:pd.DataFrame) -> pd.DataFrame:
        df.loc[df['opposite_team_side'] == 'left', 'x'] = -df.loc[df['opposite_team_side'] == 'left', 'x']
        df.loc[df['opposite_team_side'] == 'left', 'y'] = -df.loc[df['opposite_team_side'] == 'left', 'y']
        return df

    def league_average_shot_rate_per_hour_by_location(self, df:pd.DataFrame) -> pd.DataFrame:
        nb_games = df.game_id.nunique() # vu que 1 game = 1h
        shot_counts = df.groupby(['x', 'y']).size().reset_index(name='shot_count')
        shot_counts['shot_avg_league'] = shot_counts['shot_count'] / nb_games
        
        return shot_counts
    
    def team_differences(self, df:pd.DataFrame, league_shot_rate:pd.DataFrame) -> pd.DataFrame:
        shot_counts_per_team = df.groupby(['team', 'x', 'y']).size().reset_index(name='shot_count')
        nb_games_per_team = df.groupby('team')['game_id'].nunique().reset_index(name='nb_games') # or hours
        
        team_differences = pd.merge(shot_counts_per_team, nb_games_per_team, on='team', how='left')
        team_differences = pd.merge(team_differences, league_shot_rate[['x','y','shot_avg_league']],on=['x','y'], how='left')
        
        team_differences['shot_avg_team'] = team_differences['shot_count'] / team_differences['nb_games']
        team_differences['shot_avg_diff'] = team_differences['shot_avg_team'] - team_differences['shot_avg_league']
        team_differences['shot_avg_diff_percentage'] = (team_differences['shot_avg_diff'] / team_differences['shot_avg_team']) * 100
        team_differences['shot_avg_rel_diff'] = (2 * team_differences['shot_avg_diff'] / (team_differences['shot_avg_team'] + team_differences['shot_avg_league']))
        
        return team_differences
    
    def get_data_for_team(self, df:pd.DataFrame, team_name:str) -> pd.DataFrame:
        df = self.adjust_coordinates(df)
        league_shot_rate = self.league_average_shot_rate_per_hour_by_location(df)
        team_differences = self.team_differences(df, league_shot_rate)
        
        return team_differences.loc[team_differences.team == team_name]

    def lissage(self, df):
        """Je crois on va avoir besoin de scipy.stats (vu les travaux de labs)"""
        x = df[['x']]
        y = df[['y']]
        shot_avg_rel_diff = df[['shot_avg_rel_diff']]

        k = stats.gaussian_kde(np.vstack([x,y,shot_avg_rel_diff]).T)
        pass
    
    def generate_plot(self, df):
        figs = []

        for team in df['team'].unique():
            for season in df['season'].unique():
                fig = go.Figure()
                fig.add_trace(go.Contour(colorscale='Viridis'))

                fig.update_layout(
                    images=[go.layout.Image(
                        source="I CAN'T FIND THE BLOODY RINK IMAGE",
                        xref="x",
                        yref="y",
                        x=0,
                        y=100,
                        sizex=100,
                        sizey=100,
                        sizing="stretch",
                        opacity=0.5,
                        layer="below")])
                
                figs.append(fig)