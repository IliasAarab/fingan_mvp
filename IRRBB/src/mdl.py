"""Discriminator for a Generative Adversarial Network"""


## Main libs
# ------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


# In-house modules
# -----------------
from .utils.parser import Parser


class GANBase:
    def __init__(self, Pr: pd.DataFrame, dim_latent_space: int = 128):

        # Discriminator, Generator
        self.D = None
        self.G = None
        # Distributions
        self.Pr = Pr
        self.Pg = None
        self.dim_latent_space = dim_latent_space
        self.Pz = None
        # Metaschema of dataset
        self.feature_names = Pr.columns
        self.n, self.v = Pr.shape

    def init_discriminator(
        self, output_activation=tfk.activations.linear,
    ):

        # MVP with hardcoded components
        # using TFs functional API to create a Directed Acylic Graph (DAG)

        # Inputlayer
        dag_input = tfk.Input(shape=(self.v,), name="input_layer")
        # Hidden layers
        dag = tfk.layers.Dense(
            units=64, activation=tfk.layers.LeakyReLU(), name="hidden_layer_1"
        )(dag_input)
        dag = tfk.layers.Dense(
            units=32, activation=tfk.layers.LeakyReLU(), name="hidden_layer_2"
        )(dag)
        dag = tfk.layers.Dense(
            units=16, activation=tfk.layers.LeakyReLU(), name="hidden_layer_3"
        )(dag)
        # Output layer
        dag_output = tfk.layers.Dense(
            units=1, activation=output_activation, name="output_layer",
        )(dag)

        # Construct model from DAG
        self.D = tfk.Model(inputs=dag_input, outputs=dag_output, name="Discriminator")

    def init_generator(
        self, output_activation=tfk.activations.tanh,
    ):
        # MVP with hardcoded components
        # using TFs functional API to create a Directed Acylic Graph (DAG)

        # Inputlayer
        dag_input = tfk.Input(shape=(self.dim_latent_space,), name="input_layer")
        # Hidden layers
        dag = tfk.layers.Dense(
            units=64, activation=tfk.layers.LeakyReLU(), name="hidden_layer_1"
        )(dag_input)
        dag = tfk.layers.Dense(
            units=32, activation=tfk.layers.LeakyReLU(), name="hidden_layer_2"
        )(dag)
        dag = tfk.layers.Dense(
            units=16, activation=tfk.layers.LeakyReLU(), name="hidden_layer_3"
        )(dag)
        # Output layer
        dag_output = tfk.layers.Dense(
            units=self.v, activation=output_activation, name="output_layer"
        )(dag)

        # Construct model from DAG
        self.G = tfk.Model(inputs=dag_input, outputs=dag_output, name="Generator")

    def generate_data(self, distribution: str, n=100, as_df=True) -> None:

        # Either Pr, Pg, Pz

        # Get distribution
        if distribution == "Pz" and self.Pz is None:
            self.init_latent_space(n)  # init latent space
        if distribution == "Pg" and self.Pg is None:
            if self.G is None:
                self.init_generator()
            if self.Pz is None:
                self.init_latent_space(n)
            self.Pg = self.G.predict(self.Pz)
        dict_ = dict(
            Pr=[self.Pr, np.ones((n, 1))],
            Pg=[self.Pg, np.zeros((n, 1))],
            Pz=[self.Pz, np.ones((n, 1)) * 99],
        )
        X = dict_[distribution][0]
        y = dict_[distribution][1]

        # Sample randomly from chosen distribution
        idx = np.random.randint(low=0, high=len(X), size=n)
        if isinstance(X, pd.DataFrame):
            data = X.loc[idx, :]
        else:
            data = X[idx, :]

        # Convert to dataframe
        if as_df and not isinstance(X, pd.DataFrame):
            if distribution == "Pz":
                cols = [f"latent_dim_{i}" for i in range(1, self.dim_latent_space + 1)]
            else:
                cols = self.feature_names
            X = pd.DataFrame(X, columns=cols,)

        return X

    def init_latent_space(self, n: int, distribution: str = "Gaussian"):

        # Gaussian
        if distribution.lower() in ["gaussian", "normal"]:
            Pz = np.random.normal(loc=0, scale=1, size=(n, self.dim_latent_space))
        # Uniform
        elif distribution.lower() in ["uniform", "rand"]:
            Pz = np.random.random(size=(n, self.dim_latent_space))
        Pz = pd.DataFrame(
            Pz, columns=[f"latent_dim_{i}" for i in range(1, self.dim_latent_space + 1)]
        )
        self.Pz = Pz


