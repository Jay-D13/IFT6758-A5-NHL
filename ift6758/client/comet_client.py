
import os
import pickle
from comet_ml import API
from joblib import Logger


class CometMLClient:
    """
    A client for interacting with the CometML API
    """

    def __init__(self, workspace, data_dir):
        self.api = API(api_key=os.environ.get('COMET_API_KEY'))
        self.data_dir = data_dir
        self.workspace = workspace

    def get_model_list(self):
        """
        Retrieves a list of models from the CometML registry
        """
        model_list = self.api.get_registry_model_names(workspace=self.workspace)
        return model_list

    def get_model(self, model_name, logger: Logger):
        """
        Retrieves a model from the CometML registry
        """
        download_folder = os.path.join(self.data_dir, model_name)
        model_file_path = os.path.join(download_folder, 'model.pkl')

        # Download model if not already downloaded
        if not os.path.exists(download_folder):
            try:
                # Get model object from comet
                comet_model = self.api.get_model(workspace=self.workspace, model_name=model_name)

                # Retrieve latest version
                last_version = comet_model.find_versions()[0]

                logger.info(f'Downloading model {model_name} to {download_folder}')
            
                comet_model.download(version=last_version, output_folder=download_folder, expand=True)
            except:
                logger.error(f'Failed to download model {model_name} to {download_folder} from comet')
        elif os.path.exists(model_file_path):
            logger.info(f'Found existing model {model_name} at {model_file_path}')

        # Load downloaded model
        try:
            model = pickle.load(open(model_file_path, 'rb'))
            return model
        except:
            logger.error(f'Failed to load model {model_name} from {model_file_path}')
            return None