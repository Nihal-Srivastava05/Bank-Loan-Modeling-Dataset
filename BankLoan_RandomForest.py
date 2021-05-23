#References: https://www.youtube.com/watch?v=Oq1cKjR8hNo
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/logistic_regression.py

import numpy as np

from DecisionTree import DecisionTree
from collections import Counter
import pandas as pd

##subset createor
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size = n_samples, replace = True)
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    
    def __init__(self, n_trees = 100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees;
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split = self.min_samples_split, max_depth = self.max_depth, n_feats = self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        #[1111 0000 1111] --> [101 101 101 101]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        #[101 101 101 101] --> do majority vote
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        
        return np.array(y_pred)
    
##############################################################################

file_loc = "Bank_Personal_Loan_Modelling.csv"
bank_data = pd.read_csv(file_loc);

#Data Preprocessing
bank_data.set_index("ID",inplace=True);

bank_data.drop('ZIP Code',axis=1,inplace=True)

bank_data.drop('Experience',axis=1,inplace=True)

bank_data.drop('CCAvg',axis=1,inplace=True)

y = np.array(bank_data['Personal Loan'].tolist());

bank_data.drop('Personal Loan', axis=1,inplace=True)

X = np.array(bank_data.values.tolist());

from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = RandomForest(n_trees=3, max_depth=10)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print ("Random Forest classifier accuracy is: ", acc*100)
