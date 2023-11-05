import os
from comet_ml import Experiment
import pickle
from sklearn.metrics import accuracy_score

class BasicModel:
    def __init__(self, clf, exp_name='test'):
        self.clf = clf
        self.exp_name = exp_name

    def run(self, X_train, y_train, X_val, y_val, save_to=None, tags=None):
        if save_to is None:
            save_to = f'../train/exp/{self.exp_name}'
        if not os.path.exists(save_to):
            os.makedirs(save_to)

        # Setup Experiment
        exp = Experiment(
            api_key=os.environ.get('COMET_API_KEY'),
            workspace='ift6758-a5-nhl',
            project_name='milestone2'
        )
        exp.set_name(self.exp_name)
        exp.log_parameter('classifier', str(type(self.clf)))
        exp.add_tags(tags)

        # Train model
        self.clf = self.clf.fit(X_train, y_train)

        # Evaluate on val set
        y_pred = self.clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        exp.log_parameter('accuracy', accuracy)

        # Save model
        model_name = f'{self.exp_name}-clf.pkl'
        with open(save_to + model_name, 'wb') as f:
            pickle.dump(self.clf, f)
        exp.log_model(str(type(self.clf)), save_to + model_name)     

        return self.clf, y_pred