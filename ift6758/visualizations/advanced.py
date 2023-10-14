import pandas as pd
import os

class AdvancedVisualization:
    def __init__(self, data_path:str, season:int):
        self.df = pd.read_pickle(os.path.join(data_path, str(season), f'{season}.pkl'))

    def get_data_for_team(self, team_name:str) -> pd.DataFrame:
        new_df = self.df.copy()
        new_df.loc[new_df.opposite_team_side=='left','x'] = -self.df.loc[new_df.opposite_team_side=='left','x']
        new_df.loc[new_df.opposite_team_side=='left','y'] = -self.df.loc[new_df.opposite_team_side=='left','y']

        # Average for league
        nb_games = new_df.game_id.nunique()

        df_dropped = new_df.drop(['team','time','event_type','period','coordinates','shooter','goalie','shot_type','empty_net','strength','opposite_team_side'],axis=1)

        df_dropped = df_dropped.groupby(['game_id','x','y']).y.agg('count').to_frame('shot_avg').reset_index()
        df_dropped['shot_avg'] = df_dropped['shot_avg'] / nb_games

        league_avg_df = df_dropped.drop(['game_id'],axis=1).groupby(['x','y']).sum().reset_index()

        # Average for teams
        team_events_dropped = new_df.drop(['time','event_type','period','coordinates','shooter','goalie','shot_type','empty_net','strength','opposite_team_side'],axis=1)
        team_events_dropped = team_events_dropped.groupby(['team','game_id','x','y']).y.agg('count').to_frame('shot_avg').reset_index()

        nb_games_per_team = team_events_dropped.groupby(['team'])['game_id'].nunique()
        nb_games_per_team = nb_games_per_team.to_frame('game_count')

        team_events_dropped['shot_avg'] = team_events_dropped['shot_avg'] / team_events_dropped.iloc[:, 0].map(nb_games_per_team.game_count).values

        team_avg_df = team_events_dropped.drop(['game_id'],axis=1).groupby(['team','x','y']).sum().reset_index()

        # Merge League and each team
        team_avg_df = team_avg_df.merge(league_avg_df, how='outer', on=['x','y'], suffixes=('_team','_league'))
        team_avg_df['diff'] = (team_avg_df['shot_avg_team'] - team_avg_df['shot_avg_league']) / ((team_avg_df['shot_avg_team'] + team_avg_df['shot_avg_league']) / 2)

        return team_avg_df
