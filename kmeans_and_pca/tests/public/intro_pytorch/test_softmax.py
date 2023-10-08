from typing import Callable
from unittest import TestCase

import torch
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.intro_pytorch.layers import SoftmaxLayer


class TestSoftmax(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_softmax_(self, set_score: Callable[[int], None]):
        try:
            layer = SoftmaxLayer()
            x = torch.Tensor([[-10, -1, 0, 1, 10], [-20, -5, 2, 5, 7]])
            expected = torch.Tensor(
                [
                    [2.0608e-09, 1.6699e-05, 4.5392e-05, 1.2339e-04, 9.9981e-01],
                    [1.6457e-12, 5.3798e-06, 5.8997e-03, 1.1850e-01, 8.7560e-01],
                ]
            )

            actual = layer(x)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_softmax_grad(self, set_score: Callable[[int], None]):
        try:
            layer = SoftmaxLayer()
            x = torch.tensor(
                [[1.6, 10.0, 0.0, 1.2, 1.0], [2.0, 5.0, 2.5, 2.2, 1.0],],
                requires_grad=True,
            )
            expected = torch.tensor(
                [
                    [0.9989, -3.9973, 0.9998, 0.9992, 0.9994],
                    [0.7944, -3.1288, 0.6611, 0.7489, 0.9244],
                ]
            )

            loss = torch.sum(torch.log(layer(x)))  # Dummy loss
            loss.backward()

            torch.testing.assert_allclose(x.grad, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise
