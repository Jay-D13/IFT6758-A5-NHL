import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import numpy as np

class AdvancedVisualization:
    def __init__(self, data_path:str):
        self.data_path = data_path
        self.density_league_cache = {}
        self.density_team_cache = {}
    
    def load_season_data(self, season:int) -> pd.DataFrame:
        path = self.data_path.format(season=season)
        self.density_league_cache[season] = None
        self.density_team_cache[season] = {}
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
    
    def get_density_prob(self, xy_kde:np.ndarray, grid_size:int, df:pd.DataFrame, bw_size = None, isLeague = False):
        coordinates = np.vstack([df['x'], df['y']])
        
        # Remove NaN values
        coordinates = coordinates[:, ~np.isnan(coordinates).any(axis=0)]

        # Compute number of games (if league, avg. number of games multiply by number of teams)
        if isLeague:
            avg_nb_games = df.groupby(['team', 'game_id']).count().groupby('team').count()['time'].mean().round()
            nb_teams = df.team.nunique()
            nb_games = avg_nb_games * nb_teams
        else:
            nb_games = df.game_id.nunique()
        
        # Compute shot rate (number of shots divided by number of games)
        shotRate = len(df) / nb_games
        
        # Compute density and scale per grid size (100 square feet) and multiply by shot rate
        kernel = stats.gaussian_kde(coordinates, bw_method=bw_size)
        density_prob = kernel(xy_kde) 
        density_prob = grid_size * density_prob * shotRate

        return density_prob
    
    def get_data_for_team(self, df:pd.DataFrame, team_name:str, season:int) -> pd.DataFrame:
        df = self.adjust_coordinates(df)
        df = df[df['x'] > 0] # Remove shots done on the other side of the red line (too rare)
        
        # Define grid size of 100 square feet and x,y coordinates min and max for density prob estimation
        grid_size = 100
        bw_size = 0.20
        x_kde = np.linspace(0, 100, grid_size + 1)
        y_kde = np.linspace(-42.5, 42.5, grid_size + 1)
        xy_kde = np.array(np.meshgrid(x_kde, y_kde)).reshape(2, -1)

        if self.density_league_cache[season] is None:
            self.density_league_cache[season] = self.get_density_prob(xy_kde, grid_size, df, isLeague=True, bw_size=bw_size)

        team_df = df[df['team'] == team_name]
        density_prob_team = self.get_density_prob(xy_kde, grid_size, team_df, bw_size=bw_size)

        diff_df = pd.DataFrame({'diff': density_prob_team - self.density_league_cache[season]})
        return diff_df
                
    def get_plot_args(self, team_name:str, season:int, grid_size=100) -> dict:
        df = self.load_season_data(season)
        df_team = self.get_data_for_team(df, team_name, season)
        self.density_team_cache[season][team_name] = df_team
        return df_team['diff'].to_numpy().reshape((grid_size + 1, grid_size + 1), order='F')