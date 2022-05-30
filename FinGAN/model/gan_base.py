"""Baseclass to construct Generative Adversarial Networks (GANs)"""


# Main libs
from typing import Callable, Dict, Union
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

# Set up logger
from DataAnalyst.utils.logging import LoggerConfiguration

log_config = LoggerConfiguration()
log_config.set_logger(log_level="DEBUG", handler="STREAM")
logger = log_config.logger


class GANBase:
    def __init__(
        self,
        Pr: pd.DataFrame,
        dim_latent_space: int = 128,
    ) -> None:
        logger.debug("Initialize GANBase class")
        # Discriminator, Generator
        self.D: tfk.Model = None
        self.G: tfk.Model = None
        # Distributions
        self.Pr = Pr
        self.Pg = None
        self.dim_latent_space = dim_latent_space
        # Metaschema of dataset
        self.feature_names = Pr.columns
        self.n, self.v = Pr.shape

    def add_discriminator(self, D: tfk.Model) -> None:

        self.D = D

    def add_generator(self, G: tfk.Model) -> None:

        self.G = G

    def init_discriminator(
        self,
        output_activation: Callable = tfk.activations.linear,
    ) -> tfk.Model:

        # TFs functional API to create a Directed Acylic Graph (DAG)

        # Inputlayer
        dag_input = tfk.Input(shape=(self.v,), name="input_layer")
        # Hidden layers
        dag = tfk.layers.Dense(
            units=self.v * 2, activation=tfk.layers.LeakyReLU(), name="hidden_layer_1"
        )(dag_input)
        dag = tfk.layers.Dense(
            units=self.v * 4, activation=tfk.layers.LeakyReLU(), name="hidden_layer_2"
        )(dag)
        dag = tfk.layers.Dense(
            units=self.v * 8, activation=tfk.layers.LeakyReLU(), name="hidden_layer_3"
        )(dag)
        dag = tfk.layers.Dense(
            units=self.v * 4, activation=tfk.layers.LeakyReLU(), name="hidden_layer_4"
        )(dag)
        dag = tfk.layers.Dense(
            units=self.v, activation=tfk.layers.LeakyReLU(), name="hidden_layer_5"
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
        output_activation: Callable = tfk.activations.tanh,
    ) -> tfk.Model:
        """TFs functional API to create a Directed Acylic Graph (DAG)"""

        # Inputlayer
        dag_input = tfk.Input(shape=(self.dim_latent_space,), name="input_layer")
        dag = tfk.layers.BatchNormalization()(dag_input)

        # Hidden layers
        dag = tfk.layers.Dense(
            units=self.dim_latent_space * 2,
            activation=tfk.layers.LeakyReLU(),
            name="hidden_layer_1",
        )(dag)
        dag = tfk.layers.BatchNormalization()(dag_input)
        
        dag = tfk.layers.Dense(
            units=self.dim_latent_space * 4,
            activation=tfk.layers.LeakyReLU(),
            name="hidden_layer_2",
        )(dag)
        dag = tfk.layers.BatchNormalization()(dag_input)
        
        dag = tfk.layers.Dense(
            units=self.dim_latent_space * 2,
            activation=tfk.layers.LeakyReLU(),
            name="hidden_layer_3",
        )(dag)
        dag = tfk.layers.BatchNormalization()(dag_input)

        # Output layer
        dag_output = tfk.layers.Dense(
            units=self.v, activation=output_activation, name="output_layer"
        )(dag)

        # Construct model from DAG
        self.G = tfk.Model(inputs=dag_input, outputs=dag_output, name="Generator")

    def generate_data(
        self, distribution: str, n: int = 100, output_type: Union[pd.DataFrame, np.ndarray, tf.Tensor] = pd.DataFrame
    )-> Union[pd.DataFrame, np.ndarray, tf.Tensor]:

        # Generate requested data
        if distribution == 'Pr': 
            data = self.Pr.sample(n= n,)
        elif distribution == 'Pg':
            try:
                Pz = self.draw_Pz(n=n)
                data = self.G(Pz)
            except AttributeError as e:
                return "Generator is missing or ill-defined!"
        elif distribution == 'Pz':
            data = self.draw_Pz(n=n)

        # Format appropriately
        if distribution == 'Pr' and output_type == np.ndarray:
            data = data.to_numpy()
        if distribution == 'Pr' and output_type == tf.Tensor:
            data = tf.convert_to_tensor(data)
        if distribution == 'Pg' and output_type == pd.DataFrame:
            data = pd.DataFrame(data= data.numpy(), columns= self.feature_names)
        if distribution == 'Pg' and output_type == np.ndarray:
            data = data.numpy()
        if distribution == 'Pz' and output_type == tf.Tensor:
            data = tf.convert_to_tensor(data)
        if distribution == 'Pz' and output_type == pd.DataFrame:
            cols = columns=[f"latent_dim_{i}" for i in range(1, self.dim_latent_space + 1) ]
            data = pd.DataFrame(data= data, columns= cols)
        
        return data


    def init_latent_space(self, distribution: str = "Gaussian"):

        self.Pz_distribution = distribution

    def draw_Pz(self, n: int):

        if self.Pz_distribution.lower() in ["gaussian", "normal"]:
            Pz = np.random.normal(loc=0, scale=1, size=(n, self.dim_latent_space))
        elif self.Pz_distribution.lower() in ["uniform", "rand"]:
            Pz = np.random.uniform(low=-1, high=1, size=(n, self.dim_latent_space))

        return Pz


class GANmonitoring(tfk.callbacks.Callback):
    def __init__(self, theoretical_mapping: Callable = None, plot_step=100):
        super().__init__()
        self.theoretical_mapping = theoretical_mapping
        self.plot_step = plot_step

    def on_train_begin(self, logs: Dict = None):
        pass

    def on_epoch_end(
        self,
        epoch,
        logs: Dict = None,
    ):
        pass
