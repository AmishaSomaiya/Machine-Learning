from typing import Callable
from unittest import TestCase

import torch
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.intro_pytorch.layers import ReLULayer


class TestReLU(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_relu_1d(self, set_score: Callable[[int], None]):
        try:
            layer = ReLULayer()
            x = torch.Tensor([-10.0, -1.0, 0.3, 1.0, 10.2]).float()
            expected = torch.Tensor([0.0, 0.0, 0.3, 1.0, 10.2]).float()

            actual = layer(x)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_relu_2d(self, set_score: Callable[[int], None]):
        try:
            layer = ReLULayer()
            x = torch.Tensor(
                [[-10.0, -1.0, 0.3, 1.0, 10.2], [-20.2, -5.5, 2.7, 5.5, 7.6]]
            ).float()
            expected = torch.Tensor(
                [[0.0, 0.0, 0.3, 1.0, 10.2], [0.0, 0.0, 2.7, 5.5, 7.6]]
            ).float()

            actual = layer(x)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_relu_4d(self, set_score: Callable[[int], None]):
        try:
            layer = ReLULayer()
            x = torch.Tensor(
                [
                    [
                        [
                            [-3.0, 5.5, 4.0, -7.0],
                            [-2.0, -5.0, -5.0, 5.0],
                            [-1.0, 5.0, -3.0, -5.0],
                            [9.0, -6.0, 5.0, 8.0],
                        ],
                        [
                            [2.0, -6.0, -2.0, -6.0],
                            [5.0, -2.0, -8.0, -7.0],
                            [3.0, 5.0, -9.0, 0.0],
                            [-6.0, 6.0, -6.0, -8.0],
                        ],
                        [
                            [-10.0, -5.0, -6.0, -7.0],
                            [5.0, 9.0, 2.0, -9.0],
                            [8.0, -5.0, 2.0, -5.0],
                            [1.0, 2.0, -8.0, -5.0],
                        ],
                    ],
                    [
                        [
                            [7.0, 5.0, 0.0, -6.0],
                            [9.0, 9.0, -10.0, -5.0],
                            [4.0, -10.0, -2.0, 7.0],
                            [2.0, -7.0, -5.0, 8.0],
                        ],
                        [
                            [5.0, 7.0, 3.0, -8.0],
                            [-1.0, 1.0, 4.0, 8.0],
                            [9.0, 5.0, -5.0, -4.0],
                            [4.0, -3.0, 9.0, 9.0],
                        ],
                        [
                            [-8.0, -7.0, 8.0, -4.0],
                            [8.0, 8.0, -3.0, -8.0],
                            [3.0, -1.0, 3.0, -1.0],
                            [-9.0, 5.0, -10.0, 6.0],
                        ],
                    ],
                ]
            ).float()
            expected = torch.Tensor(
                [
                    [
                        [
                            [0.0, 5.5, 4.0, 0.0],
                            [0.0, 0.0, 0.0, 5.0],
                            [0.0, 5.0, 0.0, 0.0],
                            [9.0, 0.0, 5.0, 8.0],
                        ],
                        [
                            [2.0, 0.0, 0.0, 0.0],
                            [5.0, 0.0, 0.0, 0.0],
                            [3.0, 5.0, 0.0, 0.0],
                            [0.0, 6.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [5.0, 9.0, 2.0, 0.0],
                            [8.0, 0.0, 2.0, 0.0],
                            [1.0, 2.0, 0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [7.0, 5.0, 0.0, 0.0],
                            [9.0, 9.0, 0.0, 0.0],
                            [4.0, 0.0, 0.0, 7.0],
                            [2.0, 0.0, 0.0, 8.0],
                        ],
                        [
                            [5.0, 7.0, 3.0, 0.0],
                            [0.0, 1.0, 4.0, 8.0],
                            [9.0, 5.0, 0.0, 0.0],
                            [4.0, 0.0, 9.0, 9.0],
                        ],
                        [
                            [0.0, 0.0, 8.0, 0.0],
                            [8.0, 8.0, 0.0, 0.0],
                            [3.0, 0.0, 3.0, 0.0],
                            [0.0, 5.0, 0.0, 6.0],
                        ],
                    ],
                ]
            ).float()

            actual = layer(x)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise
