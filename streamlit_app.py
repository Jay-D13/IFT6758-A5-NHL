import streamlit as st
import pandas as pd
import requests
import pandas as pd
from package.ift6758.client.live_game_client import LiveGameClient
from package.ift6758.client.serving_client import ServingClient

st.title("Hockey Game Goal Prediction App")

if 'serving_client' not in st.session_state:
    st.session_state.serving_client = ServingClient()

if 'live_game_client' not in st.session_state:
    st.session_state.live_game_client = LiveGameClient()

def get_predictions(model_version, game_info):
    
    model_features = {
        'logisticregression_angle': ['angle_shot'],
        'logisticregression_distance': ['distance_goal'],
        'logisticregression_distance-angle': ['distance_goal', 'angle_shot'],
    }
    
    # filter features to match model
    X = game_info['features'][model_features[model_version]]
    
    # get predictions
    y = st.session_state.serving_client.predict(X)
    
    # concat predictions to features
    df = pd.concat([X, y], axis=1)
    
    return df

with st.sidebar:
    st.subheader("Model Configuration")

    workspace_options = ['ft6758-a5-nhl/milestone2']
    model_name_options = ['Regression']#, 'XGBoost', 'RandomForest']

    model_versions = {
        'Regression': ['logisticregression_angle', 'logisticregression_distance', 'logisticregression_distance-angle'],
        # 'XGBoost': ['v1.5'],
        # 'RandomForest': ['v3.0']
    }


    workspace = st.selectbox('Workspace', workspace_options)
    model_name = st.selectbox('Model', model_name_options)

    version_options = model_versions.get(model_name, [])
    version = st.selectbox('Version', version_options)

    if st.button('Download Model'):
        model = st.session_state.serving_client.download_registry_model(workspace, model_name, version)
        st.write('Model downloaded successfully!')

infos, stats = None, None
with st.container():
    game_id = st.text_input('Game ID', help="Enter the ID of the game you want to analyze.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Ping Game'):
            try:
                game_id = int(game_id)
            except:
                game_id = None
            
            if not game_id:
                st.warning('Please enter a valid game ID.')
            else: 
                infos = st.session_state.live_game_client.ping_game(game_id)
                x_y = get_predictions(version, infos)
                stats = st.session_state.live_game_client.get_game_stats(game_id, x_y)
    
    with col2:
        if game_id:
            if st.button('Refresh'):
                infos, stats = st.session_state.live_game_client.ping_game(game_id)
                x_y = get_predictions(version, infos)
                stats = st.session_state.live_game_client.get_game_stats(game_id, x_y)


if stats and x_y is not None:
    
    home = stats['team_names'][0]
    away = stats['team_names'][1]
    home_score = stats['score'][0]
    away_score = stats['score'][1]
    xg_home = stats['xG'][0]
    xg_away = stats['xG'][1]
    home_logo = stats['team_logos'][0]
    away_logo = stats['team_logos'][1]
    
    diff_home = home_score - xg_home
    diff_away = away_score - xg_away
    
    st.subheader(f"Game {game_id}: {home} vs {away}")
    st.write(f"Period {stats['current_period']} - {stats['time_remaining']} left")
    with st.container():
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            home_logo_response = requests.get(home_logo)
            st.image(home_logo_response.text, width=100, output_format='SVG')
        
        with col2:
            st.metric(label="Canucks xG (actual)", value=f"{xg_home} ({home_score})", delta=diff_home)
            
        with col3:
            st.metric(label="Avalanche xG (actual)", value=f"{xg_away} ({away_score})", delta=diff_away)
            
        with col4:
            away_logo_response = requests.get(away_logo)
            st.image(away_logo_response.text, width=100, output_format='SVG')

    with st.container():
        st.subheader("Data used for predictions (and predictions):")
        df = pd.concat([infos['features'], pd.DataFrame(infos['predictions'])], axis=1)
        st.dataframe(df)