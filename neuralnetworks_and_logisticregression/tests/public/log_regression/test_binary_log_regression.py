from typing import Callable
from unittest import TestCase, TestLoader, TestSuite
from unittest.mock import Mock

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.log_regression.binary_log_regression import BinaryLogReg


class TestBinaryLogReg(TestCase):
    @property
    def example_weight(self):
        return np.linspace(0, 1, 784) * np.tile([-1, 1], reps=784 // 2)

    @visibility("visible")
    @partial_credit(1)
    def test_mu_ones(self, set_score: Callable[[int], None]):
        try:
            weight, bias, X, y, expected = (
                np.ones(784),
                1.0,
                np.ones((30, 784)),
                np.ones(30),
                np.ones(30),
            )
            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.mu(X, y)

            np.testing.assert_array_almost_equal(actual, expected)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_mu_medium(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 2.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784))
            y = np.ones(30)
            expected = np.array(
                [
                    0.8825,
                    0.8843,
                    0.886,
                    0.8876,
                    0.8893,
                    0.8909,
                    0.8925,
                    0.8941,
                    0.8957,
                    0.8972,
                    0.8988,
                    0.9003,
                    0.9018,
                    0.9032,
                    0.9047,
                    0.9061,
                    0.9075,
                    0.9089,
                    0.9103,
                    0.9116,
                    0.913,
                    0.9143,
                    0.9156,
                    0.9169,
                    0.9181,
                    0.9194,
                    0.9206,
                    0.9218,
                    0.9230,
                    0.9242,
                ]
            )

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.mu(X, y)

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_loss(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 1.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784))
            y = np.tile([-1, 1], reps=15)
            expected = 1.13846

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.loss(X, y)

            np.testing.assert_almost_equal(actual, expected, decimal=4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_gradient_J_weight(self, set_score: Callable[[int], None]):
        try:
            weight = np.linspace(0, 1, num=784)
            bias = 1.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784))
            y = np.tile([-1, 1], reps=15)
            expected = np.linspace(0.23334325, 0.2519893, 784,)

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.gradient_J_weight(X, y)

            np.testing.assert_array_almost_equal(actual, expected, decimal=3)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_gradient_J_bias(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 1.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784))
            y = np.tile([-1, 1], reps=15)
            expected = 0.2777953

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.gradient_J_bias(X, y)

            np.testing.assert_almost_equal(actual, expected, decimal=4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_predict(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 0.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784)) - 0.5
            expected = np.array(
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            )

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.predict(X)

            np.testing.assert_array_almost_equal(actual, expected, decimal=3)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_misclassification_error(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 0.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784)) - 0.5
            y = np.tile([-1, 1], reps=15)
            expected = 0.467

            model = BinaryLogReg()
            model.weight = weight
            model.bias = bias

            actual = model.misclassification_error(X, y)

            np.testing.assert_array_almost_equal(actual, expected, decimal=3)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_step(self, set_score: Callable[[int], None]):
        try:
            weight = self.example_weight
            bias = 0.0
            X = np.linspace(0, 1, num=30 * 784).reshape((30, 784)) - 0.5
            y = np.tile([-1, 1], reps=15)
            expected_weight = np.copy(weight) - np.ones(784)
            expected_bias = -1.0
            learning_rate = 1.0

            model = BinaryLogReg()
            model.gradient_J_bias = Mock(return_value=1.0)
            model.gradient_J_weight = Mock(return_value=np.ones(784))
            model.weight = weight
            model.bias = bias

            model.step(X, y, learning_rate=learning_rate)

            np.testing.assert_almost_equal(model.bias, expected_bias, decimal=4)
            np.testing.assert_array_almost_equal(
                model.weight, expected_weight, decimal=4
            )
            set_score(1)
        except:  # noqa: E722
            raise


# Create a Suite for this problem
suite_binary_log_reg = TestLoader().loadTestsFromTestCase(TestBinaryLogReg)

BinaryLogRegressionTestSuite = TestSuite([suite_binary_log_reg])
