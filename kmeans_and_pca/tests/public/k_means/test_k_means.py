from typing import Callable
from unittest import TestCase

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.k_means.k_means import calculate_centers, calculate_error, cluster_data


class TestKMeans(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_calculate_centers(self, set_score: Callable[[int], None]):
        # Generate data
        X = np.array(
            [
                [0.45661771, 0.00735389, 0.53891497, 0.49414108, 0.42106833],
                [0.06777478, 0.53803311, 0.63039881, 0.35436133, 0.26452085],
                [0.4075596, 0.34979712, 0.39227966, 0.26330506, 0.34369754],
                [0.23215113, 0.56826766, 0.98684022, 0.53296418, 0.76294115],
                [0.71214384, 0.12850487, 0.35956537, 0.90565454, 0.80679593],
                [0.05206074, 0.08097557, 0.8271055, 0.27906836, 0.50902155],
                [0.93227544, 0.58239045, 0.46563451, 0.26682361, 0.53778692],
                [0.51848924, 0.11188827, 0.40287768, 0.92407813, 0.44462604],
                [0.28633286, 0.39394324, 0.91496952, 0.56278963, 0.67093319],
                [0.57184362, 0.47601387, 0.33028007, 0.95487092, 0.84502364],
            ]
        )
        c = np.array([2, 0, 0, 2, 0, 1, 2, 1, 1, 1])
        num_centers = 3

        expected = np.array(
            [
                [0.39582607, 0.33877837, 0.46074795, 0.50777364, 0.47167144],
                [0.35718162, 0.26570524, 0.61880819, 0.68020176, 0.6174011],
                [0.54034809, 0.386004, 0.66379656, 0.43130963, 0.57393213],
            ]
        )

        actual = calculate_centers(X, c, num_centers)
        np.testing.assert_almost_equal(actual, expected, decimal=4)
        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_calculate_error(self, set_score: Callable[[int], None]):
        # Generate data
        X = np.array(
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5],]
        )
        centers = np.array([[0.5, 0.0, 0.5], [0.25, 0.5, 0.25],])
        expected = 0.4183

        actual = calculate_error(X, centers)
        np.testing.assert_almost_equal(actual, expected, decimal=4)
        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_cluster_data(self, set_score: Callable[[int], None]):
        # Generate data
        X = np.array(
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5],]
        )
        centers = np.array([[0.5, 0.0, 0.5], [0.25, 0.5, 0.25],])
        expected = np.array([0, 0, 1, 1])

        actual = cluster_data(X, centers)
        np.testing.assert_almost_equal(actual, expected)
        set_score(1)
