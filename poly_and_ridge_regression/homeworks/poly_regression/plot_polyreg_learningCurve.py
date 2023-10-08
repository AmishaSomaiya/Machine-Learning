import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import learningCurve  # type: ignore
else:
    from .polyreg import learningCurve


# ----------------------------------------------------
# Plotting tools


def plotLearningCurve(errorTrain, errorTest, regLambda, degree):
    """
        plot computed learning curve
    """
    minX = 3
    maxY = max(errorTest[minX + 1 :])

    xs = np.arange(len(errorTrain))
    plt.plot(xs, errorTrain, "r-o", label="Training Error")
    plt.plot(xs, errorTest, "b-o", label="Testing Error")
    plt.plot(xs, np.ones(len(xs)), "k--")
    plt.title(f"Learning Curve (d={degree}, lambda={regLambda})")
    plt.xlabel("Training samples")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.ylim(top=maxY)
    plt.xlim((minX, 10))


def generateLearningCurve(X, y, degree, regLambda):
    """
        computing learning curve via leave one out CV
    """

    n = len(X)

    errorTrains = np.zeros((n, n - 1))
    errorTests = np.zeros((n, n - 1))

    loo = model_selection.LeaveOneOut()
    itrial = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        (errTrain, errTest) = learningCurve(
            X_train, y_train, X_test, y_test, regLambda, degree
        )

        errorTrains[itrial, :] = errTrain
        errorTests[itrial, :] = errTest
        itrial = itrial + 1

    errorTrain = errorTrains.mean(axis=0)
    errorTest = errorTests.mean(axis=0)

    plotLearningCurve(errorTrain, errorTest, regLambda, degree)


# -----------------------------------------------

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    y = allData[:, [1]]

    # generate Learning curves for different params
    plt.figure(figsize=(15, 9), dpi=100)
    plt.subplot(2, 3, 1)
    generateLearningCurve(X, y, 1, 0)
    plt.subplot(2, 3, 2)
    generateLearningCurve(X, y, 4, 1e-6)
    plt.subplot(2, 3, 3)
    generateLearningCurve(X, y, 8, 1e-6)
    plt.subplot(2, 3, 4)
    generateLearningCurve(X, y, 8, 0.1)
    plt.subplot(2, 3, 5)
    generateLearningCurve(X, y, 8, 1)
    plt.subplot(2, 3, 6)
    generateLearningCurve(X, y, 8, 100)
    plt.legend()
    plt.show()
