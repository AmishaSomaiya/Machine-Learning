# Amisha H. Somaiya
# CSE546 - HW2
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem

@problem.tag("hw2-A")
def create_synthetic_data(n , d , k , std):
    x = np.random.normal(size=(n, d))
    epsilon = np.random.normal(loc=0, scale=std, size=(n,))
    weights = np.zeros((d, 1))
    for j in range(k):
        weights[j] = (j + 1) / k
    y = np.reshape(np.dot(weights.T, x.T) + epsilon.T, (n,))
    _lambda = np.max(2 * np.abs(np.dot(y.T - np.mean(y), x)))

    return (x, y, weights, _lambda)



@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """


    return 2 * np.sum(np.square(X), axis=0)


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """

    n = X.shape[0]
    d = X.shape[1]
    c = np.zeros((d,))

    b = 1 / n * np.sum(y - np.dot(weight.T, X.T))
    wt = np.zeros((d))

    for k in range(d):

        c[k] = 2 * np.dot(X[:, k], y - (b + np.dot(weight.T, X.T) - weight[k] * X[:, k]))

        if c[k] < -1 * _lambda:
            weight[k] = (c[k] + _lambda) / a[k]
        elif c[k] > _lambda:
            weight[k] = (c[k] - _lambda) / a[k]
        else:
            weight[k] = 0

    return (weight, b)


@problem.tag("hw2-A")
def loss(X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """

    return np.sum(np.square(np.dot(X, weight) + bias - y)) + _lambda*(np.sum(np.abs(weight)))


@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-9,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """

    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None

    if start_weight is None:
        start_weight = np.zeros(X.shape[1])

    old_w = start_weight + 1 #in order that the first iteration does not converge

    while convergence_criterion(start_weight, old_w, convergence_delta) == False:
        old_w = np.copy(start_weight)
        (start_weight, b) = step(X, y , start_weight, a, _lambda)

    return (start_weight, b)


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """


    return(np.max(np.abs(weight - old_w)) < convergence_delta)


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
   
    n = 500
    d = 1000
    k = 100
    std = 1

    (X, y, weight, _lambda) = create_synthetic_data(n, d, k, std)
    lambda_factor = 2
    delta = 1e-3

    lambda_current = _lambda
    lambda_values = [_lambda]
    weight = np.zeros(d)
    (w_new, b) = train(X, y, _lambda, delta, start_weight=weight)
    W = np.zeros((d, 1))



    while np.count_nonzero(W[:, -1]) <= 999:
        lambda_current = lambda_current / lambda_factor
        lambda_values.append(lambda_current)
        print(lambda_current)
        (W_curr, bias) = train(X = X,y = y, _lambda = lambda_current, convergence_delta= delta ,start_weight = w_new)
        w_new = W_curr
        W_curr = np.reshape(W_curr, (d, 1))
        W = np.append(W, W_curr, 1)



    plt.figure(1)
    plt.xscale('log')
    plt.plot(lambda_values, np.count_nonzero(W, axis=0))
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Nonzero Coefficients of Weight Vector')
    plt.show()


    FDR = np.append([0], np.count_nonzero(W[k:, 1:], axis=0) / np.count_nonzero(W[:, 1:], axis=0))
    TPR = np.count_nonzero(W[:k, :], axis=0) / k
    plt.figure(2)
    plt.plot(FDR, TPR, "x-")
    plt.title('False Discoveries and True Positives')
    plt.xlabel('False Discovery Rate FDR')
    plt.ylabel('True Positive Rate TPR')
    plt.show()



if __name__ == "__main__":
    main()
