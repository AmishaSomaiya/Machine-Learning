from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem

# @problem.tag("hw4-A")
# def __init__(self, mu):
#     self.mu =  np.ndarray = None

@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """


    reconstructed = np.dot(demean_data, np.dot(uk,uk.T))
    return reconstructed

@problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """

    # demean_data.shape = (n,d)
    (n, d) = demean_data.shape
    reconstructed = np.dot(demean_data, np.dot(uk, uk.T))
    difference = reconstructed-demean_data
    error = np.linalg.norm(difference)**2     #, ord=2, axis=1) /n
    error = error/n
    return error

@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of it.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,). Should be in descending order.
            2. Matrix with eigenvectors as columns with shape (d, d)
    """


    n, d = demean_data.shape
    cov_matrix = np.dot(demean_data.T, demean_data)/n

    # using np.linalg.eigh since covariance matrix is symmetric
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    eig_index = np.argsort(eig_vals)[::-1]
    eig_vals_sorted, eig_vecs_sorted = eig_vals[eig_index], eig_vecs[:,eig_index]
    return (eig_vals_sorted, eig_vecs_sorted)

@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    If the handout instructs you to implement the following sub-problems, you should:

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - Plot reconstruction error as a function of k (# of eigenvectors used)
            Use k from 1 to 101.
            Plot should have two lines, one for train, one for test.
        - Plot ratio of sum of eigenvalues remaining after k^th eigenvalue with respect to whole sum of eigenvalues.
            Use k from 1 to 101.

    Part D:
        - Visualize 10 first eigenvectors as 28x28 grayscale images.

    Part E:
        - For each of digits 2, 6, 7 plot original image, and images reconstructed from PCA with
            k values of 5, 15, 40, 100.
    """
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")



    n, d = x_tr.shape
    I = np.ones((n,1))


    # self.mu = np.dot(x_tr.T, I)/n
    mu = np.dot(x_tr.T, I) / n
    demean_data = (x_tr-np.dot(I, mu.T))
    eigen_values, eigen_vectors = calculate_eigen(demean_data)

    # A4a
    print("Lambda1:", eigen_values[0])
    print("Lambda2:", eigen_values[1])
    print("Lambda10:",eigen_values[9])
    print("Lambda30:", eigen_values[29])
    print("Lambda50:" ,eigen_values[49])
    print("Sum of eigen values:", sum(eigen_values))

    # A4c
    plt.figure(figsize=(10,6))
    fig, axes = plt.subplots(3,5)
    plot_labels = [(2,5), (6,13), (7,15)]
    plot_ks = [-1, 5, 15, 40, 100]
    for row, (label, index) in enumerate(plot_labels):
        for col, k in enumerate(plot_ks):
            if k == -1:
                axes[row, col].imshow(x_tr[index].reshape(28,28))
                axes[row, col].set_title('Original Image')
                axes[row, col].axis('Off')
            else:
                reconstruct = (reconstruct_demean(eigen_vectors[:, :k], demean_data) + mu.reshape((784, )))[index]
                axes[row, col].imshow(reconstruct.reshape(28, 28))
                axes[row, col].set_title('k = {}'.format(k))
                axes[row, col].axis('Off')
    plt.savefig('A4c.png')
    # plt.imshow()
    # plt.show()
if __name__ == "__main__":
    main()
