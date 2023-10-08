from typing import Callable
from unittest import TestCase

import torch
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

from homeworks.intro_pytorch.losses import MSELossLayer


class TestMSE(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_mse_1d(self, set_score: Callable[[int], None]):
        try:
            layer = MSELossLayer()
            x = torch.Tensor([-10, -1, 0, 1, 10])
            target = torch.Tensor([-9, 2, 1, 1, 10])
            expected = 2.2

            actual = layer(x, target)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_mse_2d(self, set_score: Callable[[int], None]):
        try:
            layer = MSELossLayer()
            x = torch.Tensor([[-10, -1, 0, 1, 10], [-20, -5, 2, 5, 7]])
            target = torch.Tensor([[-12, 0.5, 0.5, 12, 9.5], [-19.5, -5.5, 1, 5, 6.7]])
            expected = 12.934

            actual = layer(x, target)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise

    @visibility("visible")
    @partial_credit(1)
    def test_mse_4d(self, set_score: Callable[[int], None]):
        try:
            layer = MSELossLayer()
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
            target = torch.Tensor(
                [
                    [
                        [
                            [-1.8705e00, 6.5835e00, 4.0499e00, -6.9518e00],
                            [-2.5455e00, -5.6398e00, -6.2703e00, 4.5112e00],
                            [-2.5713e00, 3.7789e00, -3.4891e00, -4.6798e00],
                            [9.8930e00, -6.5234e00, 5.1191e00, 7.8820e00],
                        ],
                        [
                            [1.2925e00, -6.6582e00, -3.7781e-01, -5.6831e00],
                            [5.3296e00, -2.7555e00, -8.4299e00, -7.1719e00],
                            [4.5583e00, 6.8911e00, -8.8248e00, 1.1864e00],
                            [-5.6416e00, 4.6082e00, -5.8869e00, -8.4838e00],
                        ],
                        [
                            [-1.0652e01, -5.7344e00, -5.1475e00, -7.3715e00],
                            [4.8867e00, 9.2702e00, 2.5880e00, -1.0969e01],
                            [6.9676e00, -6.3659e00, 2.2400e00, -6.3126e00],
                            [1.7163e00, 5.4429e-01, -7.0523e00, -5.0764e00],
                        ],
                    ],
                    [
                        [
                            [7.4453e00, 4.8722e00, -3.0211e-03, -6.6110e00],
                            [9.2575e00, 9.5801e00, -9.8352e00, -6.8009e00],
                            [3.6798e00, -1.0403e01, -9.4720e-01, 6.6149e00],
                            [2.1203e00, -6.3357e00, -3.7542e00, 6.8744e00],
                        ],
                        [
                            [4.6985e00, 8.4808e00, 3.1725e00, -6.9598e00],
                            [-5.1902e-01, 3.6064e00, 4.2384e00, 6.8167e00],
                            [8.3609e00, 6.0477e00, -4.3856e00, -3.1210e00],
                            [2.7550e00, -2.7739e00, 8.9919e00, 1.0666e01],
                        ],
                        [
                            [-7.7733e00, -7.6890e00, 8.6214e00, -4.1332e00],
                            [8.1093e00, 9.3485e00, -1.0028e00, -7.4632e00],
                            [3.7556e00, -2.2277e00, 3.4608e00, -1.9043e00],
                            [-9.7839e00, 4.5416e00, -8.2725e00, 5.2642e00],
                        ],
                    ],
                ]
            )
            expected = 0.863678

            actual = layer(x, target)
            torch.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)
            set_score(1)
        except:  # noqa: E722
            raise
