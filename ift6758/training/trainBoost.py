import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

class AdvancedModel():
    def __init__(self, clf):
        self.clf = clf
        
    def train(self, X_train, y_train):
        self.clf = self.clf.fit(X_train, y_train)
        return self.clf
    
    def evaluate(self, X_val, y_val):
        y_pred = self.clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return y_pred, accuracy
    
    def get_pred_proba(self, X_val):
        return self.clf.predict_proba(X_val)[:, 1]
        
    def cross_val(self, param_grid, X_train, y_train):
        kfold = RepeatedStratifiedKFold(n_splits=  5, n_repeats = 3, random_state=42)
        g_search = RandomizedSearchCV(estimator = self.clf, param_distributions = param_grid, scoring = 'accuracy', cv=kfold, n_iter = 10)
        result = g_search.fit(X_train, y_train)
        self.best_params = result.best_params_
        self.cvResults = result.cv_results_
        return self.best_params
        
    def hp_plot(self):
        with open('/Users/JJKaufman/DESS/IFT6758/IFT6758-A5-NHL/ift6758/training/figures/results.pickle', 'rb') as handle:
            cvResults = pickle.load(handle)
        
        params = cvResults['params']
        scores = cvResults['mean_test_score']
        data = [{'params': param, 'accuracy': score} for param, score in zip(params, scores)]
        data = sorted(data, key=lambda x: x['accuracy'] )
        param_vals = [d['params']for d in data]
        acc_vals = [d['accuracy'] for d in data]
        
        plt.barh(range(len(param_vals)), acc_vals)
        plt.xlim(0.909, .9099)
        plt.yticks(range(len(param_vals)), param_vals, fontsize = 6)
        plt.xticks(fontsize = 8)
        plt.ylabel('Hyperparameter Values')
        plt.xlabel('Accuracy')
        plt.title('Hyperparameter Values vs. Accuracy for XGBoost Classifier (all features)')
        #plt.tight_layout()
        plt.savefig('/Users/JJKaufman/DESS/IFT6758/IFT6758-A5-NHL/ift6758/training/figures/hyperparameters.png')
        plt.show()
        
    
    def save(self, model_name_path):
        with open(model_name_path, 'wb') as f:
            pickle.dump(self.clf, f)

   