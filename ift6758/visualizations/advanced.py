import pandas as pd
from scipy import stats
import numpy as np

class AdvancedVisualization:
    def __init__(self, data_path:str):
        self.data_path = data_path
        self.season_df = {}
        self.density_league = {}
    
    def load_season_data(self, season:int) -> pd.DataFrame:
        path = self.data_path.format(season=season)
        self.density_league[season] = None
        return pd.read_pickle(path)
    
    def adjust_coordinates(self, df:pd.DataFrame) -> pd.DataFrame:
        df.loc[df['opposite_team_side'] == 'left', 'x'] = -df.loc[df['opposite_team_side'] == 'left', 'x']
        df.loc[df['opposite_team_side'] == 'left', 'y'] = -df.loc[df['opposite_team_side'] == 'left', 'y']
        return df    
    
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
        bw_size = 0.2
        x_kde = np.linspace(df['x'].min(), df['x'].max(), grid_size + 1)
        y_kde = np.linspace(df['y'].min(), df['y'].max(), grid_size + 1)
        xy_kde = np.array(np.meshgrid(x_kde, y_kde)).reshape(2, -1)

        if self.density_league[season] is None:
            self.density_league[season] = self.get_density_prob(xy_kde, grid_size, df, isLeague=True, bw_size=bw_size)

        team_df = df[df['team'] == team_name]
        density_prob_team = self.get_density_prob(xy_kde, grid_size, team_df, bw_size=bw_size)

        diff_df = pd.DataFrame({'diff': density_prob_team - self.density_league[season]})
        return diff_df
                
    def get_plot_args(self, team_name:str, season:int, grid_size=100) -> dict:
        try:
            df = self.season_df[season]
        except KeyError:
            df = self.load_season_data(season)
            self.season_df[season] = df
            
        df_team = self.get_data_for_team(df.copy(), team_name, season)
        # Rotate grid 90 counte clockwise degrees to match the orientation of the rink
        return np.rot90(df_team['diff'].to_numpy().reshape(grid_size+1, grid_size+1), k=3)