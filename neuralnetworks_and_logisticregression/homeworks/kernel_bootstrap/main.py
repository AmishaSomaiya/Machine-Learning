from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above
        (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain
        for-loops.
            It will be called a lot, and it has to be fast for reasonable
            run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not
            necessarily with
             multiplication.
            To use it simply append .outer to function. For example: np.add.outer,
            np.divide.outer
    """

    return (np.outer(x_i, x_j) + 1)**d

@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel.
        (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above
        (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not
            necessarily  with multiplication.
            To use it simply append .outer to function. For example: np.add.outer,
            np.divide.outer
    """

    raise_to = -gamma * (np.power(np.subtract.outer(x_i, x_j), 2))
    return np.exp(raise_to)


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel
        or rbf_kernel functions. kernel_param (Union[int, float]): Gamma
        (if kernel_function is rbf_kernel) or d
        (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """

    n = x.shape[0]
    K = kernel_function(x, x, kernel_param)
    return (np.linalg.solve(K + _lambda * np.eye(n), y))  #alpha hat as per pdf


def mean_squared_error(a: np.ndarray, b: np.ndarray):
    return np.mean(np.square(np.subtract(a, b)))

@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on
        current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or
        rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d
        (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO,
        or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    average = 0
    fold_size = len(x) // num_folds
    for i in range(0, len(x), num_folds):
        x_validation = x[i:i + num_folds]
        y_validation = y[i:i + num_folds]

        x_train = np.delete(x, range(i, i + num_folds))
        y_train = np.delete(y, range(i, i + num_folds))

        alpha_validation = train(x_train, y_train, kernel_function,
                                 kernel_param, _lambda)
        kernel_matrix = kernel_function(x_validation, x_train, kernel_param)

        y_pred = np.matmul(kernel_matrix, alpha_validation)
        average += mean_squared_error(y_pred, y_validation)

    return average / fold_size

@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value
        with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda
        from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO,
        or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2
        for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than
            welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i,
        where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i,
        where i=linspace(-5, -1)
    """



    # using grid search
    lambda_possible = 10 ** (np.linspace(-5, -1))
    gamma = np.zeros((int(len(x) * (len(x) + 1) / 2), 1))
    k = 0
    for i in range(len(x)):
        for j in range(i, len(x)):
            gamma[k] = np.square(x[i] - x[j])
            k += 1
    gamma = 1 / np.median(gamma)
    lossDict = {}

    for i in range(len(lambda_possible)):
        loss = cross_validation(x, y, rbf_kernel, gamma, lambda_possible[i],
                                num_folds)
        lossDict[lambda_possible[i]] = loss

    best_lambda = min(lossDict, key=lossDict.get)
    return (best_lambda, gamma)


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value
            with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration
        sample lambda,
        d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or
        10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j)
        for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than
            welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i,
        where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i,
        where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """



    # using grid search
    lambda_possible = 10 ** (np.linspace(-5, -1))
    d = list(range(5, 26))
    lossDict = {}

    for i in range(len(lambda_possible)):
        for j in range(len(d)):
            loss = cross_validation(x, y, poly_kernel, d[j], lambda_possible[i],
                                    num_folds)
            lossDict[(lambda_possible[i], d[j])] = loss

    best_lambda, best_d = min(lossDict, key=lossDict.get)
    return (best_lambda, best_d)


@problem.tag("hw3-A", start_line=1)
def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    bootstrap_iters: int = 300,
) -> np.ndarray:
    """Bootstrap function simulation empirical confidence interval of function class.

    For each iteration of bootstrap:
        1. Sample len(x) many of (x, y) pairs with replacement
        2. Train model on these sampled points
        3. Predict values on x_fine_grid (see provided code)

    Lastly after all iterations, calculated 5th and 95th percentiles of predictions
    for each point in x_fine_point and return them.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or
        rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d
        (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        bootstrap_iters (int, optional): [description]. Defaults to 300.

    Returns:
        np.ndarray: A (2, 100) numpy array, where each row contains 5 and 95
        percentile of
        function prediction at corresponding point of x_fine_grid.

    Note:
        - See np.percentile function.
            It can take two percentiles at the same time, and take percentiles
            along specific axis.
    """
    x_fine_grid = np.linspace(0, 1, 100)




@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report
        optimal values for
         lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and
        plot predictions
        on a fine grid
        C. For both rbf and poly kernels, plot 5th and 95th percentiles
        from bootstrap
        using x_30, y_30 (using the same fine grid as in part B)
        D. Repeat A, B, C with x_300, y_300
        E. Compare rbf and poly kernels using bootstrap as described in the pdf.
        Report 5 and 95 percentiles in errors of each function.

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds,
        causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    # LOO
    lambda_polykernel, d_polykernel = poly_param_search(x_30, y_30, 1)
    lambda_rbfkernel, gamma = rbf_param_search(x_30, y_30, 1)
    # print("For A5a: for LOO CV and n = 30:")
    # print("for polynomial kernel, lambda = ",lambda_polykernel,"and d = ",d_polykernel)
    # print("for rbf kernel, lambda = ",lambda_rbfkernel,"and gamma = ",gamma)

    alpha_hatpoly = train(x_30, y_30, poly_kernel, d_polykernel, lambda_polykernel)
    alpha_hatrbf = train(x_30, y_30, rbf_kernel, gamma, lambda_rbfkernel)

    shape = alpha_hatpoly.shape[0]
    alpha_hatpoly = alpha_hatpoly.reshape(shape, 1)

    shape2 = alpha_hatrbf.shape[0]
    alpha_hatrbf = alpha_hatrbf.reshape(shape2, 1)

    x = np.linspace(0.0, 1.0, 30)
    poly_out = poly_kernel(x, x_30, d_polykernel)
    y_predpoly = np.dot(poly_out, alpha_hatpoly)
    plt.figure(1)
    plt.plot(x_30, y_30, 'x', label="Original Data")
    plt.plot(x, f_true(x), label='True f(x)', color='green')
    plt.plot(x, y_predpoly, '--x', label='widehatf(x)', color='orange')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.show()

    rbf_out = rbf_kernel(x, x_30, gamma)
    y_predicted2 = np.dot(rbf_out, alpha_hatrbf)
    plt.figure(2)
    plt.plot(x_30, y_30, 'x', label="Original Data")
    plt.plot(x, f_true(x), label='True f(x)', color='green')
    plt.plot(x, y_predicted2, '--x', label='widehatf(x)', color='orange')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.show()

    # n=300, 10 fold CV
    lambda_poly300, d_poly300 = poly_param_search(x_300, y_300, 10)
    lambda_rbf300, gamma300 = rbf_param_search(x_300, y_300, 10)
    print("For A5c: for 10fold CV and n = 300:")
    print("for polynomial kernel, lambda = ", lambda_poly300, "and d = ", d_poly300)
    print("for rbf kernel, lambda = ", lambda_rbf300, "and gamma = ", gamma300)

    alpha_ans300 = train(x_300, y_300, poly_kernel, d_poly300, lambda_poly300)
    alpha_ans2_300 = train(x_300, y_300, rbf_kernel, gamma300, lambda_rbf300)

    shape300 = alpha_ans300.shape[0]
    alpha_ans300 = alpha_ans300.reshape(shape300, 1)

    shape2_300 = alpha_ans2_300.shape[0]
    alpha_ans2_300 = alpha_ans2_300.reshape(shape2_300, 1)

    x1 = np.linspace(0.0, 1.0, 300)
    poly300 = poly_kernel(x1, x_300, d_poly300)
    y_predicted300 = np.dot(poly300, alpha_ans300)
    plt.figure(3)
    plt.plot(x_300, y_300, 'x', label="Original Data")
    plt.plot(x1, f_true(x1), label='True f(x)', color='green')
    plt.plot(x1, y_predicted300, '--x', label='widehatf(x)', color='orange')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.show()

    rbf300 = rbf_kernel(x1, x_300, gamma300)
    y_predicted2_300 = np.dot(rbf300, alpha_ans2_300)
    plt.figure(4)
    plt.plot(x_300, y_300, 'x', label="Original Data")
    plt.plot(x1, f_true(x1), label='True f(x)', color='green')
    plt.plot(x1, y_predicted2_300, '--x', label='widehatf(x)', color='orange')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
