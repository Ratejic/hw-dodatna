import numpy as np
import pandas as pd
from collections import Counter
import random
import math
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt


def all_columns(X, rand):
    return range(X.shape[1])

def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]


def random_sqrt_columns(X, rand):
    return rand.sample(range(X.shape[1]), int(np.sqrt(X.shape[1])))

class Node:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.value = value  # Leaf node value (class label)
        self.left = left  # Left subtree
        self.right = right  # Right subtree


class Tree:
    def __init__(self, rand, get_candidate_columns, min_samples):
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples
        self.root = None
        self.importance_values = None

    def build(self, X, y):
        self.root = self._build_tree(X, y)
        return self

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        if len(X) <= self.min_samples:
            return Node(value=self._get_majority_class(y))

        feature_idx = self.get_candidate_columns(X, self.rand)[0]
        best_threshold = None
        best_gini = float('inf')

        for threshold in np.unique(X[:, feature_idx]):
            left_mask = X[:, feature_idx] <= threshold

            left_y = y[left_mask]
            right_y = y[~left_mask]

            gini = self._calculate_gini(left_y, right_y)

            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold

        if best_threshold is None:
            return Node(value=self._get_majority_class(y))

        left_mask = X[:, feature_idx] <= best_threshold
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[~left_mask], y[~left_mask]

        left_subtree = self._build_tree(left_X, left_y)
        right_subtree = self._build_tree(right_X, right_y)

        self.importance_values = best_gini

        return Node(feature=feature_idx, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _calculate_gini(self, left_y, right_y):
        left_count = len(left_y)
        right_count = len(right_y)
        total_count = left_count + right_count

        gini_left = 1 - np.sum(np.square(np.bincount(left_y) / left_count))
        gini_right = 1 - np.sum(np.square(np.bincount(right_y) / right_count))

        gini = (left_count / total_count) * gini_left + (right_count / total_count) * gini_right
        #print(gini)
        return gini

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        node = self.root
        while node.value is None:
            if type(node.threshold) == type(None):
                node = node.right
            elif x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _get_majority_class(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
    
        if len(unique_classes) == 0:
            return self.root
        
        return unique_classes[np.argmax(class_counts)]
    
class RandomForest:
    def __init__(self, rand, n):
        self.rand = rand
        self.n = n
        self.trees = []
        self.importance_values = None

    def build(self, X, y):
        num_samples = len(X)
        num_features = X.shape[1]
        self.importance_values = np.zeros(num_features)
        oob_votes = np.zeros((num_samples, len(np.unique(y))), dtype=int)
        oob_counts = np.zeros(num_samples, dtype=int)

        for _ in range(self.n):
            tree = Tree(rand=self.rand, get_candidate_columns=self.get_candidate_columns, min_samples=2)
            bootstrap_indices = self.rand.choices(range(num_samples), k=num_samples)
            bootstrap_X, bootstrap_y = X[bootstrap_indices], y[bootstrap_indices]
            oob_indices = [i for i in range(num_samples) if i not in bootstrap_indices]
            tree.build(bootstrap_X, bootstrap_y)
            self._accumulate_oob_votes(tree, X, y, oob_votes, oob_counts, oob_indices)
            self.trees.append(tree)

        self._calculate_importance_values(X, y, oob_votes, oob_counts)
        return self
    
    def get_candidate_columns(self, X, rand):
        num_features = X.shape[1]
        sqrt_num_features = int(np.sqrt(num_features))
        return rand.sample(range(num_features), sqrt_num_features)
    
    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        return np.array([Counter(row).most_common(1)[0][0] for row in np.array(predictions).T])

    def _accumulate_oob_votes(self, tree, X, y, oob_votes, oob_counts, oob_indices):
        predictions = tree.predict(X[oob_indices])
        unique_labels = np.unique(y)
        for i, idx in enumerate(oob_indices):
            label_idx = np.where(unique_labels == predictions[i])[0][0]
            oob_votes[idx][label_idx] += 1
            oob_counts[idx] += 1

    def _calculate_importance_values(self, X, y, oob_votes, oob_counts):
        num_samples = len(X)
        num_features = X.shape[1]
        oob_indices = np.where(oob_counts > 0)[0]
        oob_y = y[oob_indices]
        predictions = np.argmax(oob_votes[oob_indices], axis=1)

        for feature in range(num_features):
            original_predictions = predictions[X[oob_indices, feature] == X[oob_indices, feature]]
            perturbed_predictions = predictions[X[oob_indices, feature] != X[oob_indices, feature]]
            original_error = np.mean(original_predictions != oob_y[X[oob_indices, feature] == X[oob_indices, feature]])
            perturbed_error = np.mean(perturbed_predictions != oob_y[X[oob_indices, feature] != X[oob_indices, feature]])
            if math.isnan(perturbed_error):
                perturbed_error = 1
            self.importance_values[feature] += original_error

    def importance(self):
        return self.importance_values
    
    
    
# Return misclassification rates on training and testing data
def hw_tree_full(train, test):
    # model
    t = Tree(min_samples=2, rand=random.Random(0), get_candidate_columns=random_feature)
    
    # split train
    trainX = train[0]
    trainY = train[1]

    # train model
    t.build(trainX, trainY)

    # predict
    pred_train = t.predict(trainX)

    # split test
    testX = test[0]
    testY = test[1]

    # predict
    pred_test = t.predict(testX)

    # calculate misclassification rates
    train_mis = np.mean(pred_train != trainY)
    test_mis = np.mean(pred_test != testY)

    print(train_mis, test_mis)
    print(pred_train)


    return [(0.0, train_mis), (0.0, test_mis)]


def hw_randomforests(train, test):
    # model
    rf = RandomForest(rand=random.Random(0), n=100)

    # split train
    trainX = train[0]
    trainY = train[1]

    # train model
    rf.build(trainX, trainY)

    # predict
    pred_train = rf.predict(trainX)

    # split test
    testX = test[0]
    testY = test[1]

    # predict
    pred_test = rf.predict(testX)

    # calculate misclassification rates
    train_mis = np.mean(pred_train != trainY)
    test_mis = np.mean(pred_test != testY)

    print(train_mis, test_mis)
    print(pred_train)

    return [(0.0, train_mis), (0.0, test_mis)]


if __name__=="__main__":
    # read data
    tki = pd.read_csv("tki-resistance.csv")
    
    # split data
    y = tki.iloc[:, -1]
    X = tki.iloc[:, :-1]

    # transform y to numeric data
    y = y.replace("Bcr-abl", 0)
    y = y.replace("Wild type", 1)

    # split data into train and test
    train = (X[:130], y[:130])
    test = (X[130:], y[130:])

    # change data type to numpy array
    train = (train[0].to_numpy(), train[1].to_numpy())
    test = (test[0].to_numpy(), test[1].to_numpy())

    # model
    rf = RandomForest(rand=random.Random(0), n=100)
    rfsk = RandomForestClassifier(n_estimators=100, random_state=0)

    # run model
    rf.build(train[0], train[1])
    rfsk.fit(train[0], train[1])

    # predict
    pred_train = rf.predict(train[0])
    pred_test = rf.predict(test[0])
    pred_trainsk = rfsk.predict(train[0])
    pred_testsk = rfsk.predict(test[0])

    #print(pred_trainsk)
    #rint(pred_testsk)

    # calculate misclassification rates
    train_mis = np.mean(pred_train != train[1])
    test_mis = np.mean(pred_test != test[1])
    train_missk = np.mean(pred_trainsk != train[1])
    test_missk = np.mean(pred_testsk != test[1])

    #print(train_mis, test_mis)
    #print(train_missk, test_missk)

    # build only one tree
    """ t = Tree(min_samples=2, rand=random.Random(0), get_candidate_columns=random_feature)
    t.build(train[0], train[1])
    pred_test = t.predict(test[0])
    test_mis = np.mean(pred_test != test[1])
    print("misclassification rate of one tree:")
    print(test_mis) """

    # build 100 trees
    """ rf = RandomForest(rand=random.Random(0), n=100)
    rf.build(train[0], train[1])
    pred_test = rf.predict(test[0])
    test_mis = np.mean(pred_test != test[1])
    print("misclassification rate of 100 trees:")
    print(test_mis) """

    """ mis = []
    for i in range(1, 101):
        rf = RandomForest(rand=random.Random(0), n=i)
        rf.build(train[0], train[1])
        pred_test = rf.predict(test[0])
        test_mis = np.mean(pred_test != test[1])
        mis.append(test_mis)
        print(i, test_mis)

    plt.plot(mis)
    # save image
    plt.savefig("mis.png")
    plt.show() """

    # variable importance
    rf = RandomForest(rand=random.Random(0), n=100)
    rf.build(train[0], train[1])
    importance = rf.importance()
    print("variable importance:")
    print(importance)

    """ # variable importance using sklearn
    rfsk = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True)
    rfsk.fit(train[0], train[1])
    importance = rfsk.feature_importances_
    print("variable importance using sklearn:")
    print(importance) """

