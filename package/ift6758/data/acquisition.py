import os
import requests
import pickle
from tqdm import tqdm
from package.ift6758.data import NB_MAX_REGULAR_GAMES_PER_SEASON, NHL_GAME_URL, SeasonType
    
class NHLGameData:
    def __init__(self, data_path, base_url=NHL_GAME_URL):
        self.base_url = base_url
        self.data_path = data_path
        self.data = {}
        
        os.makedirs(data_path, exist_ok=True)

    def __add__(self, other):
        new_instance = NHLGameData(data_path=self.data_path)
        new_instance.data = {**self.data, **other.data}
        return new_instance

    def _ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)
    
    def _fetch_game_from_api(self, url : str):
        """
            Fetches game data from the NHL API and returns it as a dictionary.
            
            Args:
                url (str): The URL to fetch the data from.
        """
        response = self.session.get(url)
        if response.status_code != 200:
            return None

        return response.json()
    
    def _get_from_cache(self, season: int, season_type: SeasonType) -> bool:
        """
            Retrieve games data from downloaded cache for a specific season and season type.
            Returns True if cache was found, else return False.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                season_type (SeasonType): The season type (e.g. REGULAR, PLAYOFF).
        """

        games_path = os.path.join(self.data_path, str(season))
        self._ensure_dir(games_path)
        local_file_path = os.path.join(games_path, f"{season}-{season_type.name.lower()}.pkl")

        if season not in self.data:
            self.data[season] = {}
        
        if season_type.name.lower() not in self.data[season]:
            self.data[season][season_type.name.lower()] = []

        if os.path.exists(local_file_path):
            print(f'Loading from cache file {local_file_path}')
            with open(local_file_path, 'rb') as pickle_file:
                local_data = pickle.load(pickle_file)
                self.data[season][season_type.name.lower()] = local_data
                print(f'Found {len(local_data)} {season_type.name.lower()} games for season {season}-{season+1}')
                return True
        return False
    
    def _save_to_cache(self, season: int, season_type: SeasonType):
        """
            Save games data to cache for a specific season and season type
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                season_type (SeasonType): The season type (e.g. REGULAR, PLAYOFF).
        """
        print('Saving to cache...')
        games_path = os.path.join(self.data_path, str(season))
        self._ensure_dir(games_path)
        local_file_path = os.path.join(games_path, f"{season}-{season_type.name.lower()}.pkl")

        with open(local_file_path, 'wb') as out_file:
            pickle.dump(self.data[season][season_type.name.lower()], out_file)
    
    def fetch_game(self, season : int, game_type : SeasonType, game_num : str) -> dict:
        """
            Fetches game data from the NHL API and saves it to a local file.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                game_type (SeasonType): The type of game (regular or playoff).
                game_num (str): The game number (e.g. 0001 for the first game of the season).
        """
        game_id = f"{season}{game_type.value}{game_num}"
        url = self.base_url.format(GAME_ID=game_id)
        
        return self._fetch_game_from_api(url)
                
    def fetch_playoff_games(self, season : int):
        """
            Fetches all playoff games for a given season.
            For playoff games, the 2nd digit of the specific number gives the round of the playoffs (1-4), 
            the 3rd digit specifies the matchup (out of 8), 
            and the 4th digit specifies the game (out of 7).
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
        """
        game_type = SeasonType.PLAYOFF
        if not self._get_from_cache(season, game_type):
            for round in tqdm(range(1, 5), desc=f"Downloading {game_type.name.lower()} games for season {season}-{season+1}. Current round"):
                for matchup in range(1, 9):
                    for game in range(1, 8):
                        game = self.fetch_game(season, SeasonType.PLAYOFF, f"{round:02d}{matchup}{game}")
                        if game is None:
                            break
                        self.data[season][game_type.name.lower()].append(game)
            print(f'Found {len(self.data[season][game_type.name.lower()])} {game_type.name.lower()} games for season {season}-{season+1}')
            self._save_to_cache(season, game_type)

    def fetch_regular_games(self, season : int, num_games : int):
        """
            Fetches regular games for a given season.
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                num_games (int): The number of regular games to fetch.
        """
        game_type = SeasonType.REGULAR
        if not self._get_from_cache(season, game_type):
            for game_num in tqdm(range(1, num_games + 1), desc=f"Downloading {game_type.name.lower()} games for season {season}-{season+1}"):
                game = self.fetch_game(season, game_type, f"{game_num:04d}")
                if game is None:
                    break
                self.data[season][game_type.name.lower()].append(game)

            print(f"Found {len(self.data[season][game_type.name.lower()])} {game_type.name.lower()} games for season {season}-{season+1}")
            self._save_to_cache(season, game_type)

    def fetch_season(self, season : int, regular_games : int =NB_MAX_REGULAR_GAMES_PER_SEASON):
        """
            Fetches all games for a given season.
            
            Args:
                season (int): The season year (e.g. 2019 for the 2019-2020 season).
                regular_games (int): The number of regular games to fetch.
        """
        self.session = requests.Session()
        self.fetch_regular_games(season, regular_games)
        self.fetch_playoff_games(season)
        self.session.close()