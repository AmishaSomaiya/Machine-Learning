from typing import Callable
from unittest import TestCase, TestLoader, TestSuite

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

import homeworks.ridge_regression_mnist.ridge_regression as ridge_regression


class TestRidgeRegressionMNIST(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_predict_small(self, set_score: Callable[[int], None]):
        try:
            x = np.array(
                [
                    [2.0, 3.0, 1.0, 2.0],
                    [3.0, 2.0, 2.0, 1.0],
                    [3.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 2.0],
                ],
            )
            w = np.array(
                [
                    [2.0, 1.0, 1.0, 1.0],
                    [1.0, 2.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 2.0],
                ],
            )
            expected = np.array([1, 0, 0, 2, 3])

            actual = ridge_regression.predict(x, w)
            np.testing.assert_array_almost_equal(actual, expected)
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_predict_big(self, set_score: Callable[[int], None]):
        try:
            x = np.array(
                [
                    [1.47086080, 1.72387356, 0.17856806, 0.46473236, -0.05785782],
                    [-0.45053859, -0.49651822, -1.30664898, -0.82268478, 0.50430100],
                    [-0.72728147, 0.39506313, 1.58938699, 0.10810714, 0.60942980],
                    [0.03335591, 0.20764361, 0.86493692, -1.17825034, -1.10083653],
                    [2.00280473, 1.00497888, 0.240552, 0.79007326, 1.50428679],
                    [0.18685971, 1.70283912, 0.80003864, 0.87060848, 1.03675367],
                    [-0.60914137, 1.49701155, -1.11297478, -0.25931904, -0.46696923],
                    [-0.48042269, -0.1861396, 1.96807342, -0.18312997, 0.37916892],
                    [-0.10234401, 0.49854256, -0.06261536, -1.22044529, -1.17571270],
                    [0.18291872, -0.23942408, -1.5846508, -0.24695371, 0.26855873],
                ],
            )
            w = np.array(
                [
                    [-1.78246581, 0.22102818, 0.52395953, 1.16849907],
                    [-1.08712641, 0.53883432, 0.93925320, -0.55325233],
                    [0.90991002, 0.92438768, -2.02491461, -0.10179070],
                    [-0.50916213, -0.23464117, 0.74866168, -0.26387858],
                    [0.37620393, -0.63562179, -0.77342216, -2.43477927],
                ],
            )
            expected = np.array([2, 2, 0, 3, 2, 1, 2, 0, 3, 2])

            actual = ridge_regression.predict(x, w)
            np.testing.assert_array_almost_equal(actual, expected)
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_train_plane(self, set_score: Callable[[int], None]):
        try:
            x = np.linspace([-2, -2], [2, 2], num=16)
            y = np.sum(x, axis=1, keepdims=True)
            expected = np.array([[0.9998, 0.9998]])

            actual = ridge_regression.train(x, y, _lambda=0.01)
            np.testing.assert_array_almost_equal(
                actual.squeeze(), expected.squeeze(), decimal=4
            )
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_train_high_lambda(self, set_score: Callable[[int], None]):
        try:
            x = np.linspace([-2, -2], [2, 2], num=16)
            y = np.sum(x, axis=1, keepdims=True)
            expected = np.array([[0, 0]])

            actual = ridge_regression.train(x, y, _lambda=1e6)
            np.testing.assert_array_almost_equal(
                actual.squeeze(), expected.squeeze(), decimal=4
            )
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_one_hot(self, set_score: Callable[[int], None]):
        try:
            y = np.array([2, 3, 1, 0])
            num_classes = 4
            expected = np.array(
                [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0],]
            )

            actual = ridge_regression.one_hot(y, num_classes)
            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
        except:  # noqa: E722
            raise

        set_score(1)


# Create a Suite for this problem
suite_predict = TestLoader().loadTestsFromTestCase(TestRidgeRegressionMNIST)

RidgeRegressionTestSuite = TestSuite([suite_predict])
