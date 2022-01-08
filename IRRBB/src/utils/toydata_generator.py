"""To easily generate different types of toy datasets to experiment on"""

# Main libs
# --------------
import pandas as pd
import numpy as np


class dataGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_data(nsamples=1000, ndim=1, distribution="Gaussian", **kwargs):

        # Generate data
        if distribution.lower() in ["gaussian", "normal"]:
            loc = kwargs["loc"] if "loc" in kwargs.keys() else 0
            scale = kwargs["scale"] if "scale" in kwargs.keys() else 1
            df = np.random.normal(loc, scale, size=(nsamples, ndim))

        # Reformat data
        df = pd.DataFrame(df, columns=[f"feature_{i}" for i in range(1, ndim + 1)])

        return df

