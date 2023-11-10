import pickle
from sklearn.metrics import accuracy_score

class BoostModel():
    def __init__(self, clf):
        self.clf = clf
        
    def train(self, X_train, y_train):
        self.clf = slef.clf.fit(X_train, y_train)
        return self.clf
    
    def evaluate(self, X_val, y_val):
        y_pred = self.clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return y_pred, accuracy
    
    def get_pred_proba(self, X_val):
        return self.clf.predict_proba(X_val)[:, 1]
        
    def save(self, model_name_path):
        with open(model_name_path, 'wb') as f:
            pickle.dump(self.clf, f)