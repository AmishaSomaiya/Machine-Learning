from typing import Callable
from unittest import TestCase, TestLoader, TestSuite

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.poly_regression.polyreg import (
    PolynomialRegression,
    learningCurve,
    mean_squared_error,
)


class TestPolyReg(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_polyfeatures_ones(self, set_score: Callable[[int], None]):
        try:
            X, degree, expected = np.ones((20, 1)), 3, np.ones((20, 3))
            actual = PolynomialRegression.polyfeatures(X, degree)

            np.testing.assert_array_almost_equal(actual, expected)
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_polyfeatures_twos(self, set_score: Callable[[int], None]):
        try:
            X, degree, expected = 2 * np.ones((20, 1)), 3, np.ones((20, 3)) * [2, 4, 8]
            actual = PolynomialRegression.polyfeatures(X, degree)

            np.testing.assert_array_almost_equal(actual, expected)
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_polyfeatures_fives(self, set_score: Callable[[int], None]):
        try:
            X, degree, expected = 5 * np.ones((20, 1)), 1, 5 * np.ones((20, 1))
            actual = PolynomialRegression.polyfeatures(X, degree)

            np.testing.assert_array_almost_equal(actual, expected)
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_fit_straight_line(self, set_score: Callable[[int], None]):
        try:
            degree = 4
            reg_lambda = 0
            X = np.linspace(-1, 1, 10).reshape(-1, 1)
            y = np.ones_like(X) * 2
            expected = np.array([2, 0, 0, 0, 0])

            model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
            model.fit(X, y)
            actual = (
                model.weight.squeeze()
            )  # Squeeze so that (n, ) and (n, 1) are treated the same

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_fit_linear(self, set_score: Callable[[int], None]):
        try:
            degree = 2
            reg_lambda = 0
            X = np.linspace(-1, 1, 10).reshape(-1, 1)
            y = 2 + X
            expected = np.array(
                [2, 0.6382847, 0]
            )  # The reason why second element is not one, is because of normalization

            model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
            model.fit(X, y)
            actual = (
                model.weight.squeeze()
            )  # Squeeze so that (n, ) and (n, 1) are treated the same

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(2)
    def test_fit_cubic(self, set_score: Callable[[int], None]):
        try:
            degree = 3
            reg_lambda = 1e-4
            X = np.linspace(-1, 1, 10).reshape(-1, 1)
            y = X ** 3
            expected = [
                0,
                0,
                0,
                5.003966e-01,
            ]  # The reason why this is not one, is because of normalization

            model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
            model.fit(X, y)
            actual = model.weight.squeeze()

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
        except:  # noqa: E722
            raise

        set_score(2)

    @visibility("visible")
    @partial_credit(2)
    def test_fit_hard(self, set_score: Callable[[int], None]):
        try:
            degree = 5
            reg_lambda = 1e-4
            X = np.random.RandomState(446).randn(10, 1)
            y = np.random.RandomState(546).randn(10, 1)
            expected = [
                3.6896788e-01,
                1.5476836e00,
                3.7248025e00,
                -8.5457721e00,
                -8.1189063e00,
                1.1543949e01,
            ]

            model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
            model.fit(X, y)
            actual = model.weight.squeeze()

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
        except:  # noqa: E722
            raise

        set_score(2)

    @visibility("visible")
    @partial_credit(2)
    def test_fit_and_predict_straight_line(self, set_score: Callable[[int], None]):
        try:
            degree = 4
            reg_lambda = 0
            X = np.linspace(-1, 1, 10).reshape(-1, 1)
            y = np.ones_like(X) * 2
            X_test = np.linspace(0, 4, 10).reshape(-1, 1)

            expected = np.ones_like(X_test) * 2

            model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
            model.fit(X, y)
            actual = model.predict(X_test)

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
        except:  # noqa: E722
            raise

        set_score(2)

    @visibility("visible")
    @partial_credit(2)
    def test_fit_and_predict_cubic(self, set_score: Callable[[int], None]):
        try:
            degree = 3
            reg_lambda = 0
            X = np.linspace(-1, 1, 10).reshape(-1, 1)
            y = X ** 3
            X_test = np.linspace(0, 4, 10).reshape(-1, 1)

            expected = X_test ** 3

            model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
            model.fit(X, y)
            actual = model.predict(X_test)

            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
        except:  # noqa: E722
            raise

        set_score(2)

    @visibility("visible")
    @partial_credit(1)
    def test_mean_squared_error(self, set_score: Callable[[int], None]):
        try:
            a = np.array([0.5, 0.3, 0.2, 0.1, 0.4])
            b = np.array([0.5, 0.5, 0.3, 0.2, 0.7])
            expected = 0.03
            actual = mean_squared_error(a, b)

            np.testing.assert_almost_equal(actual, expected, decimal=4)

        except:  # noqa: E722
            raise

        set_score(1)

    # Learning curve will be tested through plot submission.


# Create a Suite for this problem
suite_polyreg = TestLoader().loadTestsFromTestCase(TestPolyReg)

PolyRegTestSuite = TestSuite([suite_polyreg])
