"""Vanilla Generative Adversarial Model of Goodfellow (2014)"""

# Main libs
from typing import Callable
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import pandas as pd

# In-house modules
from .gan_base import GANBase
from ..utils.parser import Parser

# Set up logger
from DataAnalyst.utils.logging import LoggerConfiguration

log_config = LoggerConfiguration()
log_config.set_logger(log_level="DEBUG", handler="STREAM")
logger = log_config.logger


class VGAN(tfk.Model, GANBase):

    # Add to keras og init method
    def __init__(self, Pr: pd.DataFrame, dim_latent_space: int = 128, D_n=1):
        logger.debug("Initialize tfk.Model")
        super().__init__()
        GANBase.__init__(self, Pr, dim_latent_space)
        self.D_n = D_n
        self.G_optimizer = tfk.optimizers.Adam(
            learning_rate=0.0003
        )
        self.D_optimizer = tfk.optimizers.Adam(
            learning_rate=0.0004
        )
        self.G_loss = tfk.losses.BinaryCrossentropy(from_logits=True)
        self.D_loss = tfk.losses.BinaryCrossentropy(from_logits=True)
        logger.debug("Initialize VGAN")

    # Add to keras og compile method
    def compile(
        self,
        D_optimizer: Callable = None,
        G_optimizer: Callable = None,
        D_loss: Callable = None,
        G_loss: Callable = None,
    ) -> None:

        super().compile()
        if D_optimizer is not None:
            self.D_optimizer = D_optimizer
        if G_optimizer is not None:
            self.G_optimizer = G_optimizer
        if D_loss is not None:
            self.D_loss = D_loss
        if G_loss is not None:
            self.G_loss = G_loss

    # Override Tensorflow's train_step() method, this is the method that is being called for each batch when
    # running .fit(). We are going to recreate this method to define the unique way GANs are trained.
    # The batch size is undertermined and can be set when calling .fit()
    def train_step(self, Pr):

        if self.D is None or self.G is None:
            raise AttributeError(
                "Make sure to initialize both Discriminator and Generator"
            )

        if isinstance(Pr, tuple):
            Pr = Pr[0]

        # Step 1: Train D for a couple of iterations
        for _ in range(self.D_n):

            # Step 1.1:  Generate artificial samples from G in current trained state
            Pg = self.generate_data("Pg", n=Pr.shape[0], as_df=False, to_numpy=False)
            # Step 1.2: Forward propagate samples into D while taping the computations:
            # We separately forward propagate Pr and Pg so that we can easily compute the 1st and 2nd term of the first part of Eq. 3
            with tf.GradientTape() as g_tape:
                D_pred_Pr = self.D(Pr, training=True)
                D_pred_Pg = self.D(Pg, training=True)
                D_pred = tf.concat(values = [D_pred_Pr, D_pred_Pg], axis= 0)
                labels = tf.concat([tf.ones((D_pred_Pr.shape[0], 1)), tf.zeros((D_pred_Pg.shape[0], 1))], axis= 0)
                # Add (scaled) noise to labels for more stable training TODO: undo hardcoding [?]
                labels += tf.random.normal(labels.shape) * 0.1
                D_loss = self.D_loss(y_true= labels, y_pred = D_pred)

            # Step 1.3: Backpropagate into D: dD_w/dw -> gradient vector of D wrt to its weights
            dD_dw = g_tape.gradient(D_loss, self.D.trainable_variables)

            # Step 1.4: Update Ds weights
            self.D_optimizer.apply_gradients(zip(dD_dw, self.D.trainable_variables))
        
        # Step 2: Train G based on Ds feedback
        with tf.GradientTape() as g_tape:
            # Step 2.1: Forward propagate samples into D, we only need to propagate generated samples now
            Pg = self.generate_data("Pg", n=Pr.shape[0], as_df=False, to_numpy=False)
            D_pred_Pg = self.D(Pg, training=True)
            # Step 2.2: Create labels -> we use the same loss as for D but instead switch up the labels such that we 
            # effectively compute -log(D(G(z))) which is the non-saturated loss fn of Goodfellow (2014):
            # Binary cross entropy -> −y.log(D(x))−(1−y).log(1−D(G(z))) 
            # Changing labels gives -> −y.log(D(G(z)))−(1−y).log(1−D(x)) 
            # 2nd term is a cte (relative to G) and y=1 giving -> -log(D(G(z)))
            labels = tf.ones((D_pred_Pg.shape[0], 1))
            labels += tf.random.normal(labels.shape) * 0.1
            G_loss = self.G_loss(y_true= labels, y_pred = D_pred_Pg)

        # Step 2.4: Backpropagate loss through D all the way up to Gs weights -> dG_w/dw
        dG_dw = g_tape.gradient(G_loss, self.G.trainable_variables)

        # Step 2.5: Update Gs weights
        self.G_optimizer.apply_gradients(zip(dG_dw, self.G.trainable_variables))

        return {"d_loss": D_loss, "g_loss": G_loss}

