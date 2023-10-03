from enum import Enum
import requests
import os
from pathlib import Path
from ift6758.data import NB_MAX_GAMES_PER_SEASON, NHL_GAME_URL
import pickle
from tqdm import tqdm

class SeasonType(Enum):
    REGULAR = '02'
    PLAYOFF = '03'

class Season:
    def __init__(self, year: int, data_path: str, download_data = True):
        """
        Initialize a :class:`Season` object. 
        If the download_data params is set to True and no cached data was found, 
        all the season types will automatically be downloaded and saved to data_path

        Args:
            year (int): The first year of the season to retrieve, i.e. for the 2016-17
                season you'd put in 2016
            data_path (str): The file path of the desired cache pickle file
            download_data (bool): If the data must be downloaded at the initialization of the current Season object
        """
    
        self.year = str(year)
        self.data_path = Path(data_path)
        self.data = self.__load_if_cached()

        if len(self.data) == 0 and download_data:
            self.download_data()
    
    def __load_if_cached(self) -> list:
        # If data is cached, load existing data
        if os.path.exists(self.data_path):
            print(f'Loading cached data in {self.data_path}')
            with open(self.data_path, 'rb') as file_data:
                cached_data = pickle.load(file_data)
                print(f'{len(cached_data)} cached games found for year {self.year}')
                return cached_data
        
        return []

    def download_data(self):
        """
        Downloads game data for all season types contains in the enum :class:`SeasonType`
        """

        if not os.path.exists(self.data_path.parent):
            os.makedirs(self.data_path.parent)
        
        self.data = []

        session = requests.Session()
        count = 0
        for season_type in SeasonType:
            for id in tqdm(range(1, NB_MAX_GAMES_PER_SEASON+1), desc=f'Downloading games for season type {season_type.name} of year {self.year}'):
                game_id = f"{self.year}{season_type.value}{id :04d}"    
                url = NHL_GAME_URL.format(GAME_ID=game_id)
                response = session.get(url)
                if response.status_code == 200: # Status code OK
                    game_data = response.json()
                    self.data.append(game_data)
                    count += 1
        
        # Save games data in a pickle file
        print(f'Saving downloaded data in file {self.data_path}')
        with open(self.data_path, 'wb') as out_file:
            pickle.dump(self.data, out_file)
        
        print(f'{count} games found for year {self.year}')
        

