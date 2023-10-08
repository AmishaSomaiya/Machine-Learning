from typing import Callable
from unittest import TestCase

import torch
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.intro_pytorch.layers import SigmoidLayer


class TestSigmoid(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_sigmoid_1d(self, set_score: Callable[[int], None]):
        try:
            layer = SigmoidLayer()
            x = torch.Tensor([-10, -1, 0, 1, 10])
            expected = torch.Tensor(
                [4.5398e-05, 2.6894e-01, 5.0000e-01, 7.3106e-01, 9.9995e-01]
            )

            actual = layer(x)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_sigmoid_2d(self, set_score: Callable[[int], None]):
        try:
            layer = SigmoidLayer()
            x = torch.Tensor([[-10, -1, 0, 1, 10], [-20, -5, 2, 5, 7],])
            expected = torch.Tensor(
                [
                    [4.5398e-05, 2.6894e-01, 5.0000e-01, 7.3106e-01, 9.9995e-01],
                    [2.0612e-09, 6.6929e-03, 8.8080e-01, 9.9331e-01, 9.9909e-01],
                ]
            )

            actual = layer(x)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_sigmoid_4d(self, set_score: Callable[[int], None]):
        try:
            layer = SigmoidLayer()
            x = torch.Tensor(
                [
                    [
                        [
                            [-3.0, 5.0, 4.0, -7.0],
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
            )
            expected = torch.Tensor(
                [
                    [
                        [
                            [4.7426e-02, 9.9331e-01, 9.8201e-01, 9.1105e-04],
                            [1.1920e-01, 6.6929e-03, 6.6929e-03, 9.9331e-01],
                            [2.6894e-01, 9.9331e-01, 4.7426e-02, 6.6929e-03],
                            [9.9988e-01, 2.4726e-03, 9.9331e-01, 9.9966e-01],
                        ],
                        [
                            [8.8080e-01, 2.4726e-03, 1.1920e-01, 2.4726e-03],
                            [9.9331e-01, 1.1920e-01, 3.3535e-04, 9.1105e-04],
                            [9.5257e-01, 9.9331e-01, 1.2339e-04, 5.0000e-01],
                            [2.4726e-03, 9.9753e-01, 2.4726e-03, 3.3535e-04],
                        ],
                        [
                            [4.5398e-05, 6.6929e-03, 2.4726e-03, 9.1105e-04],
                            [9.9331e-01, 9.9988e-01, 8.8080e-01, 1.2339e-04],
                            [9.9966e-01, 6.6929e-03, 8.8080e-01, 6.6929e-03],
                            [7.3106e-01, 8.8080e-01, 3.3535e-04, 6.6929e-03],
                        ],
                    ],
                    [
                        [
                            [9.9909e-01, 9.9331e-01, 5.0000e-01, 2.4726e-03],
                            [9.9988e-01, 9.9988e-01, 4.5398e-05, 6.6929e-03],
                            [9.8201e-01, 4.5398e-05, 1.1920e-01, 9.9909e-01],
                            [8.8080e-01, 9.1105e-04, 6.6929e-03, 9.9966e-01],
                        ],
                        [
                            [9.9331e-01, 9.9909e-01, 9.5257e-01, 3.3535e-04],
                            [2.6894e-01, 7.3106e-01, 9.8201e-01, 9.9966e-01],
                            [9.9988e-01, 9.9331e-01, 6.6929e-03, 1.7986e-02],
                            [9.8201e-01, 4.7426e-02, 9.9988e-01, 9.9988e-01],
                        ],
                        [
                            [3.3535e-04, 9.1105e-04, 9.9966e-01, 1.7986e-02],
                            [9.9966e-01, 9.9966e-01, 4.7426e-02, 3.3535e-04],
                            [9.5257e-01, 2.6894e-01, 9.5257e-01, 2.6894e-01],
                            [1.2339e-04, 9.9331e-01, 4.5398e-05, 9.9753e-01],
                        ],
                    ],
                ]
            )

            actual = layer(x)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise
