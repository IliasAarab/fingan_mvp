"""Wasserstein Generative Adversarial Model with Gradient Penalty to ensure 1-Lipschitz continuous Discriminator"""

# Main libs
from ctypes import Union
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



class WGANGP(tfk.Model, GANBase):

    # Add to keras og init method
    def __init__(
        self, Pr: pd.DataFrame, dim_latent_space: int = 128, D_n=5, lambda_=10
    ):
        logger.debug("Initialize tfk.Model")
        super().__init__()
        GANBase.__init__(self, Pr)
        self.D_n = D_n
        self.lambda_ = lambda_
        logger.debug("Initialize WGANGP")

    # Add to keras og compile method
    def compile(self, D_optimizer, G_optimizer, D_loss, G_loss):

        super().compile()
        self.D_optimizer = D_optimizer
        self.G_optimizer = G_optimizer
        self.D_loss = D_loss
        self.G_loss = G_loss

    # Compute GP
    def _get_GP(self, Pr, Pg):

        ## Sidenote: Steps 1-5 are applied on _each_ interpolated sample but the computations have been vectorized for efficiency
        # this also mean that batchnormalization isn't allowed anymore within D, as this uses operations on the batch level whereas the GP
        # is computed on the individual sample level. Batchnorm variations can possibly be added if they do not introduce correlations between
        # individual samples.

        # Step 1: Get interpolated sample between Pr and Pg (`x_hat` of Eq.3)
        # x_hat= u*Pr + (1-u)*Pg with u~U(0,1)
        u = tf.random.uniform(shape=(Pr.shape[0], 1), minval=0, maxval=1)
        Prg = u * Pr + (1 - u) * Pg

        # Step 2: Compute Ds output based on the interpolated batch (while taping the computations)
        with tf.GradientTape() as g_tape:
            # Watch the interpolation so we can compute the gradient wrt the interpolated images (and not Ds weights!)
            # https://www.tensorflow.org/api_docs/python/tf/GradientTape
            g_tape.watch(Prg)
            # Forward propogate the interpolation through D
            D_pred_Prg = self.D(
                Prg, training=True
            )  # equivalent but faster than D.predict(Prg) unless multiple batches are used

        # Step 3: Compute the gradients of Ds weights with respect to the interpolated batch
        # so D_pred_Prg = D(Prg) and we need dD(Prg)/dPrg
        dD_dPrg = g_tape.gradient(D_pred_Prg, Prg)

        # Step 4: Compute the norm of the gradient vector
        norm_dD_dPrg = tf.sqrt(tf.reduce_sum(tf.square(dD_dPrg), axis=1))

        # Step 5: Compute the gradient penalty as described in Eq.3 of the paper
        gp = tf.reduce_mean(tf.square(norm_dD_dPrg - 1))

        return gp

    # Override Tensorflow's train_step() method, this is the method that is being called for each batch when
    # running .fit(). We are going to recreate this method to define the unique way GANs are trained.
    # The batch size is undertermined and can be set when calling .fit()
    def train_step(self, Pr):

        if isinstance(Pr, tuple):
            Pr = Pr[0]

        if isinstance(
            Pr, pd.DataFrame
        ):  # TODO: conversion must happen much earlier -> create getter for Pr ?
            logger.debug("Converting Pr to numpy ndarray")
            Pr = Pr.to_numpy().astype("float32")


        # Step 1: Train D until convergence / pre-determined number of steps
        # -----------------------------------------------------------------------------------------------------------
        for _ in range(self.D_n):  # TODO: create convergence option
            # Step 1.1:  Generate artificial samples from G in current trained state
            Pg = self.generate_data("Pg", n=Pr.shape[0], as_df=False, to_numpy=False)
            # Step 1.2: Forward propagate samples into D while taping the computations:
            # We separately forward propagate Pr and Pg so that we can easily compute the 1st and 2nd term of the first part of Eq. 3
            with tf.GradientTape() as g_tape:
                D_pred_Pr = self.D(Pr, training=True)
                D_pred_Pg = self.D(Pg, training=True)
                D_loss = self.D_loss(D_pred_Pr, D_pred_Pg)
                # Get GP
                GP = self._get_GP(Pr, Pg)
                # Compute Eq. 3
                D_loss += self.lambda_ * GP

            # Step 1.3: Backpropagate into D: dD_w/dw -> gradient vector of D wrt to its weights
            dD_dw = g_tape.gradient(D_loss, self.D.trainable_variables)

            # Step 1.4: Update Ds weights
            self.D_optimizer.apply_gradients(zip(dD_dw, self.D.trainable_variables))

        # After convergence D should approximate the one Lipschitz continous function within the Wasserstein Distance formulation

        # Step 2: Train G for one interation based on Ds feedback
        # ------------------------------------------------------------------------------------------------------------
        with tf.GradientTape() as g_tape:
            # Step 2.1: Generate artificial samples from G
            Pg = self.generate_data("Pg", n=Pr.shape[0], as_df=False, to_numpy=False)
            # Step 2.2: Forward propagate samples into D
            D_pred_Pg = self.D(Pg, training=True)
            # Step 2.3: Compute Gs loss TODO: unify D and G loss maybe
            G_loss = self.G_loss(D_pred_Pg)

        # Step 2.4: Backpropagate loss through D all the way up to Gs weights -> dG_w/dw
        dG_dw = g_tape.gradient(G_loss, self.G.trainable_variables)

        # Step 2.5: Update Gs weights
        self.G_optimizer.apply_gradients(zip(dG_dw, self.G.trainable_variables))

        return {"d_loss": D_loss, "g_loss": G_loss}
