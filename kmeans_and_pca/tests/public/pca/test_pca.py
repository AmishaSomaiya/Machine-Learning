from typing import Callable
from unittest import TestCase

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.pca.main import calculate_eigen, reconstruct_demean, reconstruction_error


class TestPCA(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_reconstruct_demean(self, set_score: Callable[[int], None]):
        X = np.array(
            [
                [-0.14154557, 0.45376136, 0.5300317, -0.59922271],
                [1.4212295, 2.23714555, -0.01203946, -1.29146967],
                [-0.17689767, 1.43306633, -0.00415335, 0.15541202],
                [0.49457332, -1.41524038, -0.10356674, 1.19404906],
                [-0.74346076, -1.4283622, 0.52023064, 1.02890164],
                [-0.85389882, -1.28037067, -0.93050279, -0.48767033],
            ]
        )

        uk = np.array(
            [
                [1.13309756, -0.46987984],
                [-0.09609833, -0.76838083],
                [0.93625666, 0.89807337],
                [-1.28469266, -0.5073952],
            ]
        )

        actual = reconstruct_demean(uk, X)
        expected = np.array(
            [
                [0.96947907, -0.4846386, 1.44152096, -1.61706724],
                [4.26700589, 1.04630673, 1.28456128, -3.02563187],
                [-0.09664994, 0.89774081, -1.49559731, 1.2543636],
                [-1.13232496, -0.03020346, -0.73470927, 1.12135914],
                [-2.39893762, -0.9216108, -0.19161864, 1.27200388],
                [-1.60857244, -0.50759469, -0.30415703, 0.9949725],
            ]
        )
        np.testing.assert_array_almost_equal(actual, expected, decimal=4)

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_reconstruction_error(self, set_score: Callable[[int], None]):
        X = np.array(
            [
                [-0.14154557, 0.45376136, 0.5300317, -0.59922271],
                [1.4212295, 2.23714555, -0.01203946, -1.29146967],
                [-0.17689767, 1.43306633, -0.00415335, 0.15541202],
                [0.49457332, -1.41524038, -0.10356674, 1.19404906],
                [-0.74346076, -1.4283622, 0.52023064, 1.02890164],
                [-0.85389882, -1.28037067, -0.93050279, -0.48767033],
            ]
        )

        uk = np.array(
            [
                [1.13309756, -0.46987984],
                [-0.09609833, -0.76838083],
                [0.93625666, 0.89807337],
                [-1.28469266, -0.5073952],
            ]
        )

        actual = reconstruction_error(uk, X)
        expected = 5.7002
        np.testing.assert_array_almost_equal(actual, expected, decimal=4)

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_calculate_eigen(self, set_score: Callable[[int], None]):
        X = np.array(
            [
                [-0.14154557, 0.45376136, 0.5300317, -0.59922271],
                [1.4212295, 2.23714555, -0.01203946, -1.29146967],
                [-0.17689767, 1.43306633, -0.00415335, 0.15541202],
                [0.49457332, -1.41524038, -0.10356674, 1.19404906],
                [-0.74346076, -1.4283622, 0.52023064, 1.02890164],
                [-0.85389882, -1.28037067, -0.93050279, -0.48767033],
            ]
        )

        actual_vals, actual_vec = calculate_eigen(X)
        expected_vals = [2.8718, 0.4877, 0.3037, 0.1279]
        expected_vec = [
            [-0.3173, -0.4797, 0.8004, -0.1689],
            [-0.8532, -0.1603, -0.3628, 0.3386],
            [-0.034, -0.4478, -0.4453, -0.7746],
            [0.4125, -0.7374, -0.1714, 0.5067],
        ]
        np.testing.assert_array_almost_equal(actual_vals, expected_vals, decimal=4)
        # Take absolute value because depending on implementation it can be up to a minus sign
        np.testing.assert_array_almost_equal(
            np.abs(actual_vec), np.abs(expected_vec), decimal=4
        )

        set_score(1)
