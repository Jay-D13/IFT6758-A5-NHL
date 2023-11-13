from comet_ml import API
import os 
import pickle
from ift6758.features import FeatureEng
from ift6758.training.plot import plot_all

workspace='ift6758-a5-nhl'
out_folder = './'
model_file_name = 'model.pkl'
clean_data_path = './ift6758/data/json_clean/'
models_dict_regular = {}
models_dict_playoff = {}

# Get Test data
feature_eng = FeatureEng(clean_data_path)
test_data = feature_eng.getTestSet(2020)
test_data = feature_eng.encodeCategories(test_data, ['shot_type', 'prev_type'])
test_data = test_data.rename(columns = {'shot_type_Slap Shot':'shot_type_Slap_Shot', 'shot_type_Snap Shot':'shot_type_Snap_Shot', 'shot_type_Wrist Shot':'shot_type_Wrist_Shot'})
test_data_regular = test_data[test_data['game_id'].astype(str).str.contains('\d\d\d\d02\d\d\d\d', regex=True)]
test_data_playoff = test_data[test_data['game_id'].astype(str).str.contains('\d\d\d\d03\d\d\d\d', regex=True)]

X_test_regular = test_data_regular.drop(['is_goal', 'game_id'], axis=1)
X_test_playoff = test_data_playoff.drop(['is_goal', 'game_id'], axis=1)
y_test_regular = test_data_regular['is_goal']
y_test_playoff = test_data_playoff['is_goal']

# Connect to the comet API
api = API(api_key=os.environ.get('COMET_API_KEY'))

# Retrieve registered model list
model_list = api.get_registry_model_names(workspace=workspace)

for model_name in model_list:
    # Get model object from comet
    comet_model = api.get_model(workspace=workspace, model_name=model_name)

    # Retrieve latest version
    last_version = comet_model.find_versions()[0]
    print(f'Found version {last_version} for model {model_name}')

    # Download model
    download_folder = os.path.join(out_folder, model_name)
    comet_model.download(version=last_version, output_folder=download_folder, expand=True)
    model_file_path = os.path.join(download_folder, model_file_name)

    # Load model
    clf = pickle.load(open(model_file_path, 'rb'))

    # Get features to keep
    experiment_key = comet_model.get_details(last_version)['experimentKey']
    tags = api.get_experiment_by_key(experiment_key).get_tags()

    if 'AllFeatures' in tags:
        X_test_selected_regular = X_test_regular
        X_test_selected_playoff = X_test_playoff
    elif 'distance_goal' in tags and 'angle_shot' in tags:
        X_test_selected_regular = X_test_regular[['distance_goal', 'angle_shot']]
        X_test_selected_regular = X_test_selected_regular.rename(columns={'distance_goal': 'distance', 'angle_shot': 'angle'})
        X_test_selected_playoff = X_test_playoff[['distance_goal', 'angle_shot']]
        X_test_selected_playoff = X_test_selected_playoff.rename(columns={'distance_goal': 'distance', 'angle_shot': 'angle'})
    elif 'distance_goal' in tags:
        X_test_selected_regular = X_test_regular['distance_goal'].values.reshape(-1, 1)
        X_test_selected_playoff = X_test_playoff['distance_goal'].values.reshape(-1, 1)
    elif 'angle_shot' in tags:
        X_test_selected_regular = X_test_regular['angle_shot'].values.reshape(-1, 1)
        X_test_selected_playoff = X_test_playoff['angle_shot'].values.reshape(-1, 1)
    else:
        X_test_selected_regular = X_test_regular
        X_test_selected_playoff = X_test_playoff

    # Evaluate model on test set
    y_pred_prob_regular = clf.predict_proba(X_test_selected_regular)[:, 1]
    y_pred_prob_playoff = clf.predict_proba(X_test_selected_playoff)[:, 1]

    models_dict_regular[model_name] = y_pred_prob_regular
    models_dict_playoff[model_name] = y_pred_prob_playoff

plot_all(models_dict_regular, y_test_regular, save_to_folder=os.path.join(out_folder, 'TestRegular'))
plot_all(models_dict_playoff, y_test_playoff, save_to_folder=os.path.join(out_folder, 'TestPlayoff'))