import unittest
import numpy as np
import random

# from hw_tree import Tree, RandomForest, hw_tree_full, hw_randomforest
from hw_tree import Tree, RandomForest, hw_tree_full, hw_randomforests

def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]

class HWTreeTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 1, 1, 1, 0, 1, 1, 1])
        self.train = self.X[:6], self.y[:6]
        self.test = self.X[6:], self.y[6:]

    def test_call_tree(self):
        t = Tree(rand=random.Random(1),
                 get_candidate_columns=random_feature,
                 min_samples=2)
        p = t.build(*self.train)
        pred = p.predict(self.test[0])
        # print(f"X: {self.test[0]}")
        # print(f"Expected: {self.test[1]}")
        # print(f"Prediction: {pred}")
        np.testing.assert_equal(pred, self.test[1])

    def test_call_randomforest(self):
        rf = RandomForest(rand=random.Random(1),
                          n=10)
        p = rf.build(*self.train)
        pred = p.predict(self.test[0])
        np.testing.assert_equal(pred, self.test[1])

    def test_call_importance(self):
        rf = RandomForest(rand=random.Random(1),
                          n=10)
        p = rf.build(*self.train)
        imp = p.importance()
        print(f"calculated importance: {imp}")
        expected_importance = [0.4, 0.6]  # Replace with the expected values
        np.testing.assert_allclose(imp, expected_importance, rtol=1e-05)

    def test_hw_tree_full(self):
        (train_misclass, test_misclass) = hw_tree_full(self.train, self.test)
        expected_train_misclass = 0.0  # Replace with the expected values
        expected_test_misclass = 0.0  # Replace with the expected values
        self.assertAlmostEqual(train_misclass, expected_train_misclass)
        self.assertAlmostEqual(test_misclass, expected_test_misclass)

    def test_hw_randomforest(self):
        (train_misclass, test_misclass) = hw_randomforests(self.train, self.test)
        expected_train_misclass = 0.0  # Replace with the expected values
        expected_test_misclass = 0.0  # Replace with the expected values
        self.assertAlmostEqual(train_misclass, expected_train_misclass)
        self.assertAlmostEqual(test_misclass, expected_test_misclass)


if __name__ == "__main__":
    unittest.main()

