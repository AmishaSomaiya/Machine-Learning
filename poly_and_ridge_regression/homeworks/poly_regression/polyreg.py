"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1E-8):
        """Constructor
        """
        # __init__
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        self.weight: np.ndarray = None  # type: ignore
        # To use the learnt mean and std deviation of training data for predicting the test data
        self.means: np.array = None
        self.stds: np.array = None


    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """

        # polyfeatures(X, degree)
        n = len(X)
        rows = n
        cols = degree
        mat_output = np.zeros((rows, cols))

        # To expand the given X into an (n, degree) array of polynomial features of degree degree
        for i in range(n):
            z = 0
            while z <= degree - 1:
                mat_output[i][z] = X[i] ** (z+1)
                z += 1
        return mat_output

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """


        # To expand the given X into an (n, degree) array of polynomial features of degree degree
        mat_output = self.polyfeatures(X, self.degree)
        self.means = np.mean(mat_output, axis = 0)
        self.stds = np.std(mat_output, axis = 0)
        n, d = mat_output.shape
        # Standardization
        for i in range(d):
            mat_output[:,i] = (mat_output[:, i] - self.means[i]) / self.stds[i]

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), mat_output]
        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0
        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y)




    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """


        # Expanding Xtest to polynomial features
        mat_output = self.polyfeatures(X, self.degree)
        n, d = mat_output.shape
        # Standardizing test data using learnt mean and std deviation from fit()
        for i in range(d):
            mat_output[:,i] = (mat_output[:, i] - self.means[i] ) / self.stds[i]
        # add 1s column
        X_ = np.c_[np.ones([n, 1]), mat_output]
        return X_ @ self.weight

@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """

    # Finding mean squared error for arrays a and b
    n = len(a)
    temp = 0.0
    for i in range(n):
        temp += (a[i] - b[i])**2
    return temp/n



@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # The PolyRegression model:
    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    # For errorTrain:
    for i in range(n):
        model.fit(Xtrain[0: i+1], Ytrain[0: i+1])
        yPredTrain = model.predict(Xtrain[0: i+1])
        errorTrain[i] = (mean_squared_error(yPredTrain, Ytrain[0: i+1]))
    # For errorTest:
    for i in range(n):
        model.fit(Xtrain[0: i+1], Ytrain[0: i+1])
        yPredTest = model.predict(Xtest[0: i + 1])
        errorTest[i] = (mean_squared_error(yPredTest, Ytest[0: i + 1]))

    return [errorTrain, errorTest]

