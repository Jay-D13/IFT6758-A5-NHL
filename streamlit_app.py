import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import pandas as pd
from ift6758.ift6758.client.live_game_client import LiveGameClient
from ift6758.ift6758.client.serving_client import ServingClient

st.title("Hockey Game Goal Prediction App")

with st.sidebar:
    st.subheader("Model Configuration")
    workspace = st.text_input('Workspace') # maybe we can make a dropdown with the available workspaces, model names, and versions
    model_name = st.text_input('Model')
    version = st.text_input('Version')
    if st.button('Download Model'):
        model = ServingClient.download_registry_model(workspace, model_name, version)


with st.container():
    game_id = st.text_input('Game ID', help="Enter the ID of the game you want to analyze.")
    if st.button('Query Game'):
        X, idx, _ = LiveGameClient.ping_game(game_id)
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=json.loads(X.to_json())
        )
        response = response.json()

if 'response' in locals():
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Period", response["period"])
            st.metric("Time Remaining", response["time_remaining"])

        with col2:
            st.metric("Home Team", response["team_names"][0])
            st.metric("Away Team", response["team_names"][1])

        with col3:
            st.metric("Current Score", response["current_score"])

            xg_sum = response["xG"].sum()
            score_diff = xg_sum - response["current_score"]
            st.metric("xG Score Difference", f"{score_diff:.2f}")

    with st.container():
        st.subheader("Data used for predictions (and predictions):")
        st.dataframe(response)

if __name__ == '__main__':
    st.run()