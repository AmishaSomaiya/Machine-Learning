from typing import Callable
from unittest import TestCase

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.kernel_bootstrap.main import poly_kernel, rbf_kernel


class TestKernelBootstrap(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_rbf_kernel_same(self, set_score: Callable[[int], None]):
        # Generate data
        x_i = np.array([0.2, -0.5, 0.3, -0.4])
        gamma = 5.0
        expected = np.array(
            [
                [1.0000, 0.0863, 0.9512, 0.1653],
                [0.0863, 1.0000, 0.0408, 0.9512],
                [0.9512, 0.0408, 1.0000, 0.0863],
                [0.1653, 0.9512, 0.0863, 1.0000],
            ]
        )

        try:
            actual = rbf_kernel(x_i, x_i, gamma)
        except:  # noqa: E722
            raise

        np.testing.assert_almost_equal(actual, expected, decimal=4)
        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_rbf_kernel_different(self, set_score: Callable[[int], None]):
        # Generate data
        x_i = np.array([0.2, -0.5, 0.3, -0.4])
        x_j = np.array([-0.1, 0.2, 0.6, -0.9])
        gamma = 5.0
        expected = np.array(
            [
                [6.3763e-01, 1.0000e00, 4.4933e-01, 2.3579e-03],
                [4.4933e-01, 8.6294e-02, 2.3579e-03, 4.4933e-01],
                [4.4933e-01, 9.5123e-01, 6.3763e-01, 7.4659e-04],
                [6.3763e-01, 1.6530e-01, 6.7379e-03, 2.8650e-01],
            ]
        )

        try:
            actual = rbf_kernel(x_i, x_j, gamma)
        except:  # noqa: E722
            raise

        np.testing.assert_almost_equal(actual, expected, decimal=4)
        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_poly_kernel_same(self, set_score: Callable[[int], None]):
        # Generate data
        x_i = np.array([0.2, -0.5, 0.3, -0.4])
        d = 4
        expected = np.array(
            [
                [1.1699, 0.6561, 1.2625, 0.7164],
                [0.6561, 2.4414, 0.522, 2.0736],
                [1.2625, 0.522, 1.4116, 0.5997],
                [0.7164, 2.0736, 0.5997, 1.8106],
            ]
        )

        try:
            actual = poly_kernel(x_i, x_i, d)
        except:  # noqa: E722
            raise

        np.testing.assert_almost_equal(actual, expected, decimal=4)
        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_poly_kernel_different(self, set_score: Callable[[int], None]):
        # Generate data
        x_i = np.array([0.2, -0.5, 0.3, -0.4])
        x_j = np.array([-0.1, 0.2, 0.6, -0.9])
        d = 4
        expected = np.array(
            [
                [0.9224, 1.1699, 1.5735, 0.4521],
                [1.2155, 0.6561, 0.2401, 4.4205],
                [0.8853, 1.2625, 1.9388, 0.284],
                [1.1699, 0.7164, 0.3336, 3.421],
            ]
        )

        try:
            actual = poly_kernel(x_i, x_j, d)
        except:  # noqa: E722
            raise

        np.testing.assert_almost_equal(actual, expected, decimal=4)
        set_score(1)
