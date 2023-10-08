"""
    Sample implementation of linear regression using direct computation of the solution
    AUTHOR Eric Eaton
"""

import numpy as np

"""
-----------------------------------------------------------------
 Class LinearRegression - Closed Form Implementation
-----------------------------------------------------------------
"""


class LinearRegressionClosedForm:
    def __init__(self, reg_lambda=1e-8):
        """
        Constructor
        """
        self.regLambda = reg_lambda
        self.theta = None

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-d array
                y is an n-by-1 array
            Returns:
                No return value
        """
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]

        n, d = X_.shape
        # remove 1 for the extra column of ones we added to get the original num features
        d = d - 1

        # construct reg matrix
        reg_matrix = self.regLambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y)

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]

        # predict
        return X_.dot(self.theta)


"""
-----------------------------------------------------------------
 End of Class LinearRegression - Closed Form Implementation
-----------------------------------------------------------------
"""
