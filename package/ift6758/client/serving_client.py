import requests
import pandas as pd
import logging
import os
import json


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self):
        
        ip = os.environ.get('SERVING_IP')
        port = os.environ.get('SERVING_PORT', '5002')
        self.base_url = f"http://{ip}:{port}"

        print(f"Initializing client; base URL: {self.base_url}")
        
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
        X = X.to_json(orient='split', index=False)
        res = requests.post(url, json=X)
        if res.status_code == 200:
            response_data = res.json()
            predictions = pd.DataFrame(response_data['predictions'])

            predictions.columns = ['not_goal', 'goal']

            return predictions
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
        data = {"workspace": workspace, "model": model, "version": version}
        rappelle = requests.post(url, json=data)
        if rappelle.status_code ==200:
            modele_donnees = rappelle.json()
            return modele_donnees
        else:
            logger.error(f"Download registry model request failed with status code {rappelle.status_code}")
            rappelle.raise_for_status()            




        