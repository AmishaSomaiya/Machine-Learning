if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    If the handout instructs you to implement the following sub-problems, you should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")

    total_classes = 10
    all_centres, difference = lloyd_algorithm(x_train, total_classes)
    all_centres = all_centres.reshape((10,28,28))

    count = 0
    fig, axis = plt.subplots(2,5)
    for coord1 in range(2):
        for coord2 in range(5):
            axis[coord1, coord2].imshow(all_centres[count, : :])
            # axis[coord1, coord2].set_title("For centre", count)
            axis[coord1, coord2].set_title(f"For centre {count}")
            axis[coord1, coord2].axis("Off")
            count +=1

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
