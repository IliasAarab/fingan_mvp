"""Discriminator for a Generative Adversarial Network"""


## Main libs
# ------------------
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


# In-house modules
# -----------------
from ..utils.parse import Parse

# Set up logger
from DataAnalyst.utils.logging import LoggerConfiguration

log_config = LoggerConfiguration()
log_config.set_logger(log_level="DEBUG", handler="STREAM")
logger = log_config.logger

# What is the problem? two problems actually:
# 1. we need to be able to make flexible Gs and Ds
# 2. we need to be ablt to have a generic set of methods and attributes

# 1. We can make a function that adapts a subclassed layer, but this would still be within a base class


class GANBase:
    def __init__(
        self,
        Pr: pd.DataFrame,
        dim_latent_space: int = 128,
    ):
        logger.debug("Initialize GANBase class")
        # Discriminator, Generator
        self.D = None
        self.G = None
        # Distributions
        self.Pr = Pr
        self.Pg = None
        self.dim_latent_space = dim_latent_space
        # Metaschema of dataset
        self.feature_names = Pr.columns
        self.n, self.v = Pr.shape

    def init_discriminator(
        self,
        output_activation=tfk.activations.linear,
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
            units=1,
            activation=output_activation,
            name="output_layer",
        )(dag)

        # Construct model from DAG
        self.D = tfk.Model(inputs=dag_input, outputs=dag_output, name="Discriminator")

    def init_generator(
        self,
        output_activation=tfk.activations.tanh,
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

    def generate_data(
        self, distribution: str, n=100, as_df=True, to_numpy=True
    ) -> None:

        # Either Pr or Pg

        # for Pg
        if distribution == "Pg":
            if self.G is None:
                self.init_generator()
            if self.__Pz_distribution is None:
                self.init_latent_space(distribution="Gaussian")
            Pz = self.draw_Pz(n=n)
            if to_numpy:
                data = self.G(Pz).numpy()
            else:
                data = self.G(Pz)
            y = np.zeros((n, 1))

        # for Pr
        elif distribution == "Pr":
            # Sample randomly from Pr dataset
            idx = np.random.randint(low=0, high=len(self.Pr), size=n)
            if isinstance(self.Pr, pd.DataFrame):
                data = self.Pr.loc[idx, :].copy()
            else:
                data = self.Pr[idx, :]
            y = np.ones((n, 1))

        # Convert to dataframe
        if as_df and not isinstance(data, pd.DataFrame):
            cols = self.feature_names
            data = pd.DataFrame(
                data,
                columns=cols,
            )

        return data

    def init_latent_space(self, distribution: str = "Gaussian"):

        self.__Pz_distribution = distribution

    def draw_Pz(self, n: int):

        # Gaussian
        if self.__Pz_distribution.lower() in ["gaussian", "normal"]:
            Pz = np.random.normal(loc=0, scale=1, size=(n, self.dim_latent_space))
        # Uniform
        elif self.__Pz_distribution.lower() in ["uniform", "rand"]:
            Pz = np.random.random(size=(n, self.dim_latent_space))

        #  if as_df: #redundant?
        #     Pz = pd.DataFrame(
        #         Pz,
        #         columns=[
        #             f"latent_dim_{i}" for i in range(1, self.dim_latent_space + 1)
        #         ],
        #     )

        return Pz
