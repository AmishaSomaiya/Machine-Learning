# Amisha H. Somaiya
# CSE546 HW2
if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    ytrain = df_train["ViolentCrimesPerPop"]
    y_train = ytrain.to_numpy()
    xtrain = df_train.drop("ViolentCrimesPerPop", axis=1)
    x_train = xtrain.to_numpy()

    ytest = df_test["ViolentCrimesPerPop"]
    y_test = ytest.to_numpy()
    xtest = df_test.drop("ViolentCrimesPerPop", axis=1)
    x_test = xtest.to_numpy()

    delta = 1e-5  # threshold used to determine when to stop searching for optimal w
    d = x_train.shape[1]

    _lambda = np.max(2*np.abs(np.dot(y_train.T - np.mean(y_train), x_train)))
    lambda_factor = 2
    lambda_current = _lambda
    lambda_values = [_lambda]

    weight = np.zeros(d)
    W = np.zeros((d, 1))

    # agePct12t29, pctWSocSec, pctUrban, agePct65up, and householdsize
    agePct12t29 = xtrain.columns.get_loc("agePct12t29")
    pctWSocSec = xtrain.columns.get_loc("pctWSocSec")
    pctUrban = xtrain.columns.get_loc("pctUrban")
    agePct65up = xtrain.columns.get_loc("agePct65up")
    householdsize = xtrain.columns.get_loc("householdsize")

    # Initialize vectors to plot A6d
    wt_agePct12t29 = []
    wt_pctWSocSec = []
    wt_pctUrban = []
    wt_agePct65up = []
    wt_householdsize = []
    trainerror = []
    testerror = []

    (w_new, b) = train(x_train, y_train, _lambda, delta, weight)

    while lambda_current > 0.01:
        # w_new = 0
        (W_curr, bias) = train(X=x_train, y=y_train, _lambda=lambda_current, convergence_delta=delta,
                               start_weight=w_new)
        lambda_current = lambda_current / lambda_factor
        lambda_values.append(lambda_current)
        print(lambda_current)
        trainerror.append(np.mean((x_train.dot(W_curr) + bias - y_train) ** 2))
        testerror.append(np.mean((x_test.dot(W_curr) + bias - y_test) ** 2))

        w_new = np.copy(W_curr)
        W_curr = np.reshape(W_curr, (d, 1))
        W = np.append(W, W_curr, 1)
        wt_agePct12t29.append(W_curr[agePct12t29])
        wt_pctWSocSec.append(W_curr[pctWSocSec])
        wt_pctUrban.append(W_curr[pctUrban])
        wt_agePct65up.append(W_curr[agePct65up])
        wt_householdsize.append(W_curr[householdsize])


     # A6c
    plt.figure(3)
    plt.xscale('log')
    plt.plot(lambda_values, np.count_nonzero(W, axis=0))
    plt.xlabel('Lambda')
    plt.ylabel('Nonzero Coefficients of Weight Vector')
    plt.show()

    # A6d
    del lambda_values[len(lambda_values) - 1]
    plt.plot(lambda_values, wt_agePct12t29, "x-")
    plt.xscale('log')
    plt.plot(lambda_values, wt_pctWSocSec, "x-")
    plt.xscale('log')
    plt.plot(lambda_values, wt_pctUrban, "x-")
    plt.xscale('log')
    plt.plot(lambda_values, wt_agePct65up, "x-")
    plt.xscale('log')
    plt.plot(lambda_values, wt_householdsize, "x-")
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Weights')
    plt.title('Regularization Paths for Lambda')
    plt.legend(['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize'])
    plt.show()

    # A6e
    plt.plot(lambda_values, trainerror)
    plt.plot(lambda_values, testerror)
    plt.xscale('log')
    plt.legend(['Train Error', 'Test Error'])
    plt.title('MSE error for Lambda')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Mean Squared Error MSE')
    plt.show()


    # For A6f, lambda = 30
    strt_wt_30 = np.zeros(d)
    (w_new, b) = train(x_train, y_train, 30, delta, strt_wt_30)
    (W_30, bias) = train(X=x_train, y=y_train, _lambda=30, convergence_delta=delta,
                         start_weight=w_new)
    print("largest (most positive) Lasso coefficient", xtrain.columns[np.argmax(W_30)], max(W_30))
    print("Smallest (most negative) Lasso coefficient", xtrain.columns[np.argmin(W_30)], min(W_30))







if __name__ == "__main__":
    main()
