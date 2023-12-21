import requests
import pandas as pd
import logging
import os


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self):
        
        ip = os.environ.get('SERVING_IP')
        port = os.environ.get('SERVING_PORT')
        
        self.base_url = f"http://{ip}:{port}"
        
        
        logger.info(f"Initializing client; base URL: {self.base_url}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        url = f"{self.base_url}/predict"
        res = requests.post(url, json = X.to_json())
        if res.status_code == 200:
            df = pd.DataFrame(res.json())
            return df
            
        else:
            logger.error(f"Prediction request failed with status code {res.status_code}")
            res.raise_for_status()
            
    def logs(self) -> dict:
        """Get server logs"""
        url = f"{self.base_url}/logs"
        rappelle = requests.get(url)
        if rappelle.status_code == 200:
            logs = rappelle.json()
            return logs
        else:
            logger.error(f"Logs request failed with status code {rappelle.status_code}")
            rappelle.raise_for_status()
        
    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        url = f"{self.base_url}/download_registry_model"
        params = {"workspace": workspace, "model": model, "version": version}
        rappelle = requests.get(url, params=params)
        if rappelle.status_code ==200:
            modele_donnees = rappelle.json()
            return modele_donnees
        else:
            logger.error(f"Download registry model request failed with status code {rappelle.status_code}")
            rappelle.raise_for_status()            




        