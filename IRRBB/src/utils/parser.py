"""Parses data into the right format"""


# main libs
# ------------
import pandas as pd
import numpy as np

# Typehinting
# -----------
from typing import Union

from pandas.core.algorithms import isin
from pandas.core.frame import DataFrame


class Parser:
    def __init__(self):
        pass

    @staticmethod
    def _atleast2D(v: np.ndarray) -> np.ndarray:
        """Makes sure that a vector is always a column vector with dimensions (n, v) \
            with n=samples, v=variables"""
        if len(v.shape) == 1:
            v = np.atleast_2d(v).T
        else:
            pass
        return v

    @staticmethod
    def to_two_dfs(
        data: Union[pd.DataFrame, "tuple[np.ndarray, np.ndarray]"], **kwargs
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """A GANs discriminator/critic ingests/outputs the dataset pair (X,y) consists of:
        - df: dataframe.
        - X: matrix with Xjk the kth attribute of the jth sample (real or generated)
        - y: vector with 0 = real sample and 1 = generated sample if y is from the training\test set \
            else if y_pred then vector with probability of samples being from the real dataset.
        
        The dataset pair will be parsed into two distinct dataframes of respectively \
            real and generated data and their attributes."""

        # Determine whether data is Pandas dataframe or tuple of X,y numpy matrices
        if isinstance(data, pd.DataFrame):
            X, y = Parser._from_one_df_to_ml_tuple(data)
            Pr, Pg = Parser._from_ml_tuple_to_two_dfs(data=(X, y), **kwargs)
        else:
            X, y = data
            Pr, Pg = Parser._from_ml_tuple_to_two_dfs(data=(X, y), **kwargs)

        return Pr, Pg

    @staticmethod
    def to_ml_tuple(
        data: Union[pd.DataFrame, "tuple[pd.DataFrame, pd.DataFrame]"]
    ) -> "tuple[np.ndarray, np.ndarray]":

        if isinstance(data, tuple):
            df = Parser._from_two_dfs_to_one_df(data)
            X, y = Parser._from_one_df_to_ml_tuple(df)
        else:
            X, y = Parser._from_one_df_to_ml_tuple(data)

        return X, y

    @staticmethod
    def to_one_df(
        data: Union[
            "tuple[pd.DataFrame, pd.DataFrame]", "tuple[np.ndarray, np.ndarray]"
        ],
        **kwargs,
    ) -> pd.DataFrame:

        if isinstance(data[0], np.ndarray):
            Pr, Pg = Parser._from_ml_tuple_to_two_dfs(data, **kwargs)
            df = Parser._from_two_dfs_to_one_df((Pr, Pg))
        else:
            df = Parser._from_two_dfs_to_one_df(data)

        return df

    @staticmethod
    def _from_two_dfs_to_one_df(
        data: "tuple[pd.DataFrame, pd.DataFrame]",
    ) -> pd.DataFrame:
        # TODO: we assume first element of tuple is Pr and 2nd is Pg
        Pr, Pg = data
        Pr, Pg = Pr.copy(), Pg.copy()
        Pr, Pg = Parser._enforce_array_same_len(data=(Pr, Pg))
        Pr["Dataset"], Pg["Dataset"] = "Pr", "Pg"
        df = pd.concat((Pr, Pg), axis=0, ignore_index=True)
        return df

    @staticmethod
    def _from_ml_tuple_to_two_dfs(
        data: "tuple[np.ndarray, np.ndarray]",
        hard_classes: bool = True,
        feature_names: "list[str]" = None,
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        X, y = data  # assume first elm is X and 2nd is y
        # convert row vector into col vector for univariate datasets
        X = Parser._atleast2D(X)
        y = Parser._atleast2D(y)

        # Step 1: Convert to hard classes 0 or 1
        if hard_classes:
            y = np.where(y > 0.5, 1, 0)[:, 0].reshape(-1, 1)

        # Step 2: Create matrix [X, y]
        m = np.hstack((X, y))
        # Step 3: Create dataFrames for each class
        Pr = m[np.where(m[:, -1] > 0.5)][:, :-1]
        Pg = m[np.where(m[:, -1] <= 0.5)][:, :-1]
        Pr, Pg = Parser._atleast2D(Pr), Parser._atleast2D(Pg)
        if feature_names is not None:
            cols = [f"feature_{i}" for i in range(1, Pg.shape[1] + 1)]
        else:
            cols = feature_names
        Pr = pd.DataFrame(Pr, columns=cols)
        Pg = pd.DataFrame(Pg, columns=cols)

        return (Pr, Pg)

    @staticmethod
    def _from_one_df_to_ml_tuple(data: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray]":

        data = data.copy().to_numpy()
        X, y = data[:, :-1], data[:, -1]
        y = np.where(y == "Pr", 1, 0)
        X = np.float16(X)
        y = np.int0(y)
        return X, y

    @staticmethod
    def _enforce_array_same_len(
        data: Union[
            "tuple[pd.DataFrame, pd.DataFrame]", "tuple[np.ndarray, np.ndarray]"
        ]
    ) -> Union["tuple[pd.DataFrame, pd.DataFrame]", "tuple[np.ndarray, np.ndarray]"]:

        Pr, Pg = data
        if isinstance(Pr, pd.DataFrame):
            cols = Pr.columns
            Pr = Pr.copy().to_numpy()
            Pg = Pg.copy().to_numpy()
        else:
            cols = None
            Pr = Pr.copy()
            Pg = Pg.copy()

        if len(Pr) != len(Pg):
            # Which one is longest
            longest = Pr if len(Pr) > len(Pg) else Pg
            smallest = Pr if len(Pr) < len(Pg) else Pg
            # Pad the smallest with NaNs to get same length as longest
            padding = len(longest) - len(smallest)
            smallest = np.float16(smallest)
            smallest = np.pad(
                smallest,
                pad_width=[(0, padding), (0, 0)],  # only pad first dim
                mode="constant",
                constant_values=np.nan,
            )
            Pr = longest if len(Pr) > len(Pg) else smallest
            Pg = longest if len(Pg) > len(Pr) else smallest

        # Edits
        Pr = np.float16(Pr)
        Pg = np.float16(Pg)
        if cols is None:
            return Pr, Pg
        Pr = pd.DataFrame(Pr, columns=cols)
        Pg = pd.DataFrame(Pg, columns=cols)
        return Pr, Pg
