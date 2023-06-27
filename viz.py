import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Generate a random dataset
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([0, 0, 1, 1])


# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=20, random_state=0)
rf.fit(X, y)

print(X.shape[1])