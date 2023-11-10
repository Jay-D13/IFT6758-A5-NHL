import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from comet_ml import Experiment
import os
from ift6758.training.trainBoost import boostModel

def main(opts):
    # Create train folder
    if not os.path.exists(os.path.join(opts.exp_path, opts.exp_name)):
        os.makedirs(os.path.join(opts.exp_path, opts.exp_name))

    # Get Data
    train_val = pd.read_pickle(opts.data_path)
    X_all = train_val.drop(['is_goal'], axis=1)[opts.use_features]
    y_all = train_val['is_goal']

    # Split into train val
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

    # Setup Experiment
    exp = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        workspace='ift6758-a5-nhl',
        project_name='milestone2'
    )
    exp.set_name(opts.exp_name)
    tags = opts.use_features
    tags.append('XGBoost')
    tags.append('AdvancedModel')
    exp.add_tags(tags)

    # Train model
    model = boostModel()
    model.train(X_train, y_train)

    # Evaluate model
    y_pred, accuracy = model.evaluate(X_val, y_val)
    exp.log_metric('accuracy', accuracy)

    # Save model
    model_path = os.path.join(opts.exp_path, opts.exp_name, 'model.pkl')
    model.save(model_path)
    exp.log_model('Model', model_path)

    exp.end()
def parse_opts(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= './ift6758/features/trainVal/TrainValSets.pk', help='Data path that contains train and val data')
    parser.add_argument('--exp_path', type=str, default= './train/', help='Experience path of parent folder')
    parser.add_argument('--exp_name', type=str, default= 'exp', help='Experience name for comet ml')
    parser.add_argument('--use_features', nargs='+', type=str, default= '[distance]', help='Feature to train XGBoostClassifier with')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    opts = parse_opts(True)
    for k, v in kwargs.items():
        setattr(opts, k, v)
    main(opts)
    return opts

if __name__ == '__main__':
    opts = parse_opts()
    main(opts)