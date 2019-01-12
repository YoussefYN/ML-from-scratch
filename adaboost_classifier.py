# This question is about AdaBoost algorithm.
# You should implement it using library function (DecisionTreeClassifier) as a base classifier
# Don't do any additional imports, everything is already there
#
# There are two functions you need to implement:
#      (a) fit
#      (b) predict


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class BoostingTreeClassifier:

    def __init__(self, random_state):
        self.random_state = random_state
        self.classifiers = []
        self.tree_weights = None

    # TO DO ---- 10 POINTS ---------- Implement the fit function ---------------------
    def fit(self, X, y, n_trees):
        self.classifiers = [DecisionTreeClassifier(random_state=self.random_state)
                            for i in range(n_trees)]
        m = len(y)
        self.tree_weights = np.zeros(shape=(n_trees,))
        alphas = np.full(shape=(m,), fill_value=1.0 / m)
        """Trains n_trees classifiers based on AdaBoost algorithm - i.e. applying same
        model on samples while changing their weights. You should only use library
        function DecisionTreeClassifier as a base classifier, the boosting algorithm
        itself should be written from scratch. Store trained tree classifiers
        in self.classifiers. Calculate tree weight for each classifier and store them
        in self.tree_weights. Initialise DecisionTreeClassifier with self.random_state

        :param X: train data
        :param y: train labels
        :param n_trees: number of trees to train
        :return: doesn't return anything
        """
        for i in range(n_trees):
            self.classifiers[i].fit(X=X, y=y, sample_weight=alphas)
            pred = self.classifiers[i].predict(X)
            error = 1 - accuracy_score(y_pred=pred, y_true=y, sample_weight=alphas)
            w = np.log((1 - error) / error) / 2.0
            self.tree_weights[i] = w
            alphas[y != pred] = alphas[y != pred] * np.exp(w)
            alphas[y == pred] = alphas[y == pred] * np.exp(-w)
            alphas = alphas / np.sum(alphas)

    # TO DO ---- 5 POINTS ---------- Implement the predict function ---------------------
    def predict(self, X):
        """Makes final predictions aggregating predictions of trained classifiers

        :param X: test data
        :return: predictions
        """
        pred = np.average(np.array([est.predict(X) for est in self.classifiers]),
                          axis=0,
                          weights=self.tree_weights)
        return np.where(pred >= 0, 1, -1)


# loading and pre-processing titanic data set
titanic = pd.read_csv('titanic_modified.csv').dropna()
data = titanic[['Pclass', 'Age', 'SibSp', 'Parch']].values
labels = titanic.iloc[:, 6].values
# changing labels so that we can apply boosting
labels[np.argwhere(labels == 0)] = -1
# splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)

# setting constants
rand_state = 3
T = 10  # number of trees

# measuring accuracy using one decision tree
tree = DecisionTreeClassifier(random_state=rand_state)
tree.fit(X_train, y_train)
print('One tree accuracy:', accuracy_score(tree.predict(X_test), y_test))

# measuring accuracy using an ensemble based on boosting
ensemble = BoostingTreeClassifier(random_state=rand_state)
ensemble.fit(X_train, y_train, T)
print('Ensemble accuracy:', accuracy_score(ensemble.predict(X_test), y_test))
