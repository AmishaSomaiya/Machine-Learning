from typing import Callable
from unittest import TestCase, TestLoader, TestSuite

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.vanilla_vs_numpy.vanilla_vs_numpy import numpy_solution, vanilla_solution


class TestVanillaVsNumpy(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_numpy_solution(self, set_score: Callable[[int], None]):
        RNG = np.random.RandomState(seed=546)
        n = 5

        # Generate some data
        x = RNG.randn(n)
        y = RNG.randn(n)
        A = RNG.randn(n, n)
        B = RNG.randn(n, n)

        a = numpy_solution(x, y, A, B)

        np.testing.assert_array_almost_equal(
            a, [5.449948, 3.881913, 3.423083, -1.525936, 4.596191], decimal=4
        )
        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_vanilla_solution(self, set_score: Callable[[int], None]):
        RNG = np.random.RandomState(seed=546)
        n = 5

        # Generate some data as python lists
        x = RNG.randn(n).tolist()
        y = RNG.randn(n).tolist()
        A = RNG.randn(n, n).tolist()
        B = RNG.randn(n, n).tolist()

        a = vanilla_solution(x, y, A, B)

        np.testing.assert_array_almost_equal(
            a, [5.449948, 3.881913, 3.423083, -1.525936, 4.596191], decimal=4
        )
        set_score(1)
