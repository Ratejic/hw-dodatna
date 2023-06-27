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

class TreeNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.value = value  # Leaf node value (class label)
        self.left = left  # Left subtree
        self.right = right  # Right subtree

    def __str__(self):
        if self.value is not None:
            return "Leaf node" + str(self.value)
        return f"Not a leaf node, feature: {self.feature}, threshold: {self.threshold}, {'has left and right' if self.left and self.right else 'missing one or both child nodes'}"


class Tree:
    def __init__(self, rand, get_candidate_columns, min_samples):
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples
        self.root = None

    def build(self, X, y):
        self.root = self._build_tree(X, y)
        self.X = X
        self.y = y
        return self

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return TreeNode(value=y[0])

        if len(X) < self.min_samples:
            return TreeNode(value=self._get_majority_class(y))

        feature_idxs = self.get_candidate_columns(X, self.rand)
        best_threshold = None
        best_feature = None
        best_gini = 10
        #print(f"feture_idxs: {feature_idxs}")
        for feature in feature_idxs:
            #print(f"tresholds: {np.unique(X[:, feature])}")
            for threshold in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= threshold

                left_y = y[left_mask]
                right_y = y[~left_mask]

                gini = self._calculate_gini(left_y, right_y)

                if gini < best_gini and len(left_y) > 0 and len(right_y) > 0:
                    best_gini = gini
                    best_threshold = threshold
                    best_feature = feature


        if best_feature is None:
            return TreeNode(value=self._get_majority_class(y))

        left_mask = X[:, best_feature] <= best_threshold
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[~left_mask], y[~left_mask]

        """ left_subtree = self._build_tree(left_X, left_y)
        right_subtree = self._build_tree(right_X, right_y) """

        levii, desnii = self.razdeli(X[:, best_feature], best_threshold)
        left_subtree = self._build_tree(X[levii, :], y[levii])
        right_subtree = self._build_tree(X[desnii, :], y[desnii])

        if not best_feature:
            pass
            #print(f"best feature: {best_feature}, best threshold: {best_threshold}, left subtree: {left_subtree}, right subtree: {right_subtree}")
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def razdeli(self, X, threshold):
        levi = np.argwhere(X <= threshold)
        levi = levi.flatten()
        desni = np.argwhere(X > threshold)
        desni = desni.flatten()
        return levi, desni


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
        self.X = X
        self.y = y
        num_samples = len(X)
        num_features = X.shape[1]
        self.importance_values = np.zeros(num_features)

        for _ in range(self.n):
            tree = Tree(rand=self.rand, get_candidate_columns=self.get_candidate_columns, min_samples=2)
            bootstrap_indices = self.rand.choices(range(num_samples), k=num_samples)
            bootstrap_X, bootstrap_y = X[bootstrap_indices], y[bootstrap_indices]
            tree.build(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

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

    def importance(self):
        num_features = self.X.shape[1]
        importances = np.zeros(num_features)
        num_samples = len(self.y)

        for tree in self.trees:
            y_pred = tree.predict(self.X)
            accuracy = np.mean(self.y == y_pred)

            for feature_idx in range(num_features):
                X_perm = np.copy(self.X)
                perm_indices = np.random.permutation(num_samples)
                X_perm[:, feature_idx] = X_perm[perm_indices, feature_idx]
                y_pred_perm = tree.predict(X_perm)
                accuracy_perm = np.mean(self.y == y_pred_perm)

                importances[feature_idx] += accuracy - accuracy_perm

        return importances / len(self.trees)
        
    
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

    train_succ = 1 - train_mis
    test_succ = 1 - test_mis
    return [(train_succ, train_mis), (test_succ, test_mis)]


def hw_randomforests(train, test):
    # model
    rf = RandomForest(rand=random.Random(0), n=100)

    # split train
    trainX = train[0]
    trainY = train[1]

    # train model
    rf.build(trainX, trainY)

    print(rf.trees)

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

    train_succ = 1 - train_mis
    test_succ = 1 - test_mis
    return [(train_succ, train_mis), (test_succ, test_mis)]


if __name__=="__main__":
    """ X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    y = np.array([0, 0, 1, 1])
    train = X[:3], y[:3]
    test = X[3:], y[3:]
    print(hw_randomforests(train, test)) """
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
    
    
    """
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

    print(train_mis, test_mis)
    print(train_missk, test_missk)

    # build only one tree
    t = Tree(min_samples=2, rand=random.Random(0), get_candidate_columns=random_feature)
    t.build(train[0], train[1])
    pred_test = t.predict(test[0])
    test_mis = np.mean(pred_test != test[1])
    print("misclassification rate of one tree:")
    print(test_mis)

    # build 100 trees
    rf = RandomForest(rand=random.Random(0), n=100)
    rf.build(train[0], train[1])
    pred_test = rf.predict(test[0])
    test_mis = np.mean(pred_test != test[1])
    print("misclassification rate of 100 trees:")
    print(test_mis)

    mis = []
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
    plt.show()
    """
    # variable importance
    rf = RandomForest(rand=random.Random(0), n=100)
    rf.build(train[0], train[1])
    importance = rf.importance()
    # normalize importance on 1-100 scale
    importance = importance / np.max(importance) * 100.0

    # plot importance
    plt.plot(importance)
    # save image
    #plt.savefig("importance.png")
    #plt.show()
    # biggest importance
    b = np.argsort(importance)[-1]
    #print(b)
    """
    # variable importance using sklearn
    rfsk = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True)
    rfsk.fit(train[0], train[1])
    importancesk = rfsk.feature_importances_
    # plot importance
    plt.plot(importancesk)
    # save image
    plt.savefig("importancesk.png")
    plt.show()
    # biggest importance
    bsk = np.argsort(importancesk)[-1]
    print(bsk) """

    rf = RandomForest(rand=random.Random(0), n=100)
    rf.build(train[0], train[1])
    features = []
    for tree in rf.trees:
        root = tree.root
        features.append(root.feature)
    
    plt.scatter(features, np.arange(1, 101), color="red")
    plt.savefig("features.png")
    plt.show()

