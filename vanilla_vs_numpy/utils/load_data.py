import numpy as np

from .sst_problem_util import SST2Dataset
from .lower_level_utils import get_homeworks_path
from pathlib import Path

def load_dataset(dataset: str, small: bool = False):
    # First check current directory
    data_path: Path = get_homeworks_path() / "data"

    if data_path is None:
        print("Could not find dataset. Please run from within 446 hw folder.")
        exit(0)

    if dataset.lower() == "mnist":
        with np.load(data_path / "mnist" / "mnist.npz", allow_pickle=True) as f:
            X_train, labels_train = f['x_train'], f['y_train']
            X_test, labels_test = f['x_test'], f['y_test']

        # Reshape each image of size 28 x 28 to a vector of size 784
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)

        # Pixel values are integers from 0 to 255.
        # Dividing the pixel values by 255 is a technique called normalization,
        # which is a standard practice in Machine Learning to prevent large numeric values
        X_train = X_train / 255
        X_test = X_test / 255

        return ((X_train, labels_train), (X_test, labels_test))

    if dataset.lower() == "polyreg":
        f = open((data_path / "polyreg" / "polydata.dat"), "r")
        allData = np.loadtxt(f, delimiter=",")
        f.close()
        return allData

    if dataset.lower() == "crime":
        import pandas as pd
        df_train = pd.read_table(data_path / "crime-data" / "crime-train.txt")
        df_test = pd.read_table(data_path / "crime-data" / "crime-test.txt")
        return df_train, df_test

    if dataset.lower() == "xor":
        return (
            (np.load(data_path / "xor" / "x_train.npy"), np.load(data_path / "xor" / "y_train.npy")),
            (np.load(data_path / "xor" / "x_val.npy"), np.load(data_path / "xor" / "y_val.npy")),
            (np.load(data_path / "xor" / "x_test.npy"), np.load(data_path / "xor" / "y_test.npy")),
        )

    if dataset.lower() == "kernel_bootstrap":
        if (data_path / "kernel_bootstrap" / "x_30.npy").exists() and (data_path / "kernel_bootstrap" / "x_300.npy").exists() and (data_path / "kernel_bootstrap" / "x_1000.npy").exists():
            return (
                (np.load(data_path / "kernel_bootstrap" / "x_30.npy"), np.load(data_path / "kernel_bootstrap" / "y_30.npy")),
                (np.load(data_path / "kernel_bootstrap" / "x_300.npy"), np.load(data_path / "kernel_bootstrap" / "y_300.npy")),
                (np.load(data_path / "kernel_bootstrap" / "x_1000.npy"), np.load(data_path / "kernel_bootstrap" / "y_1000.npy")),
            )
        else:
            # Generate arrays
            rng = np.random.RandomState(seed=2021)
            f_true = lambda x: 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)
            for n in [30, 300, 1000]:
                x = rng.rand(n)
                y = f_true(x) + rng.randn(n)
                np.save(data_path / "kernel_bootstrap" / f"x_{n}.npy", x)
                np.save(data_path / "kernel_bootstrap" / f"y_{n}.npy", y)

            return load_dataset("kernel_bootstrap")  # Load numpy arrays
    if dataset.lower() == "sst-2":
        train_dataset = SST2Dataset(data_path / "SST-2" / "train.tsv")
        val_dataset = SST2Dataset(data_path / "SST-2" / "dev.tsv", train_dataset.vocab, train_dataset.reverse_vocab)

        return train_dataset, val_dataset