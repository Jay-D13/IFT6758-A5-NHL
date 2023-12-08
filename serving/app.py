"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import logging
from flask import Flask, jsonify, request, Response
import pandas as pd
import json


from ift6758.client.comet_client import CometMLClient


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


app = Flask(__name__)

# Global variables
model = None
workspace='ift6758-a5-nhl'
model_list = None
data_dir = './download'
default_model = 'logisticregression_distance'
comet_api = None

# Setup basic logging configuration
logging.basicConfig(filename=LOG_FILE, encoding='utf-8', level=logging.INFO)

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """    
    global model, comet_api

    # Initialization before the first request
    comet_api = CometMLClient(workspace=workspace, data_dir=data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    model = comet_api.get_model(model_name=default_model, logger=app.logger)

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    with open(LOG_FILE, 'r', encoding='utf-8') as file:
        log_data = file.read().splitlines()
    
    response = {
        'log_data': log_data
    }
    
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    global model

    # Get POST json data
    json_request = request.get_json()
    app.logger.info(json_request)


    new_model = comet_api.get_model(model_name=json_request['model'], logger=app.logger)

    if new_model is not None:
        model = new_model
        response = {
            "message": "Model loaded successfully"
        }
        status_code = 201
    else:
        response = {
            "message": "Failed to load model, keeping previously loaded model"
        }
        status_code = 500

    app.logger.info(response)
    return Response(json.dumps(response), status_code, mimetype='application/json') # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    global model

    # Get POST json data
    json_data = request.get_json()
    app.logger.info(json_data)

    # Load JSON data into a DataFrame
    if isinstance(json_data[list(json_data.keys())[0]], float):
        df = pd.DataFrame(json_data, index=[0])
    else:
        df = pd.DataFrame(json_data)

    # Perform predictions using the loaded model
    predictions = model.predict_proba(df)

    response = {
        'predictions': predictions.tolist()
    }

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!