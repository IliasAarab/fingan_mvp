"""Parses data into the right format"""


# main libs
# ------------
import pandas as pd
import numpy as np


class Parser:
    def __init__(self):
        pass

    @staticmethod
    def _atleast2D(v):
        """Makes sure that a vector is always a column vector with dimensions (n, v) \
            with n=samples, v=variables"""
        if len(v.shape) == 1:
            v = np.atleast_2d(v).T
        else:
            pass
        return v

    @staticmethod
    def parse_dataset(X, y):
        """A GANs discriminator/critic ingests/outputs the dataset pair (X,y) consists of:
        - X: matrix with Xjk the kth attribute of the jth sample (real or generated)
        - y: vector with 0 = real sample and 1 = generated sample if y is from the training\test set \
            else if y_pred then vector with probability of samples being from the real dataset.
        
        The dataset pair will be parsed into two distinct dataframes of respectively \
            real and generated data and their attributes."""

        # convert row vector into col vector for univariate datasets
        X = Parser._atleast2D(X)
        y = Parser._atleast2D(y)

        # Step 1: Convert to hard classes 0 or 1
        y = np.where(y > 0.5, 1, 0)[:, 0].reshape(-1, 1)

        # Step 2: Create matrix [X, y]
        m = np.hstack((X, y))
        # Step 3: Create dataFrames for each class
        Pg = m[np.where(m[:, -1] == 0)][:, :-1]
        Pg = Parser._atleast2D(Pg)
        Pr = m[np.where(m[:, -1] == 1)][:, :-1]
        Pr = Parser._atleast2D(Pr)
        # TODO: allow to use user col names
        Pg = pd.DataFrame(
            Pg, columns=[f"feature_{i}" for i in range(1, Pg.shape[1] + 1)]
        )
        Pr = pd.DataFrame(
            Pr, columns=[f"feature_{i}" for i in range(1, Pr.shape[1] + 1)]
        )

        return (Pr, Pg)
