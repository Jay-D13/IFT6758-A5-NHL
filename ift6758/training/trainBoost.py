import pickle
from sklearn.metrics import accuracy_score
import xgboost as xgb

class boostModel():
    def __init__(self, params = None, num_boosts = 100):
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': 0.3,
                'max_depth': 3
            }
        self.params = params
        self.num_rounds = num_rounds
        self.clf = None
        
    def train(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label = y_train)
        self.clf = xgb.train(self.params, dtrain, self.num_boosts)
        
    
    def predict(self, X_test):
        if self.clf=None:
            raise ValueError("Model has not been trained. Call the 'train' method first.")
        dtest = xgb.Dmatrix(X_test)
        return self.clf.predict(dtest)

    def evaluate(self, X_val, y_val):
        y_pred = self.clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return y_pred, accuracy

    def save(self, model_name_path):
        with open(model_name_path, 'wb') as f:
            pickle.dump(self.clf, f)