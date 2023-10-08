from typing import Callable
from unittest import TestCase

import torch
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.intro_pytorch.layers import LinearLayer


class TestLinear(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_linear_shapes(self, set_score: Callable[[int], None]):
        try:
            layer = LinearLayer(20, 10)
            x = torch.ones((5, 20))

            actual = layer(x)
            torch.testing.assert_allclose(actual.shape, (5, 10))
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_linear_generator(self, set_score: Callable[[int], None]):
        try:
            generator = torch.Generator()
            generator.manual_seed(446)
            layer = LinearLayer(5, 2, generator=generator)
            x = torch.Tensor(
                [
                    [1.4, 1.2, 0.2, -0.4, 0.5],
                    [-1.4, 0.2, 0.1, -0.2, 2.5],
                    [-1.6, 1.8, 1.0, -1.8, 2.1],
                ]
            )

            actual = layer(x)
            expected_weight_first = torch.Tensor(
                [[-1.6013, -2.7034], [-3.1342, -5.6708], [-1.2706, -7.3891]]
            )
            expected_bias_first = torch.Tensor(
                [[-0.0121, -0.2437], [-2.9210, -3.5440], [-0.7473, -0.1933],]
            )
            try:
                torch.testing.assert_allclose(
                    actual, expected_weight_first, rtol=1e-2, atol=1e-4
                )
            except AssertionError:
                torch.testing.assert_allclose(
                    actual, expected_bias_first, rtol=1e-2, atol=1e-4
                )
            set_score(1)
        except:  # noqa: E722
            raise
