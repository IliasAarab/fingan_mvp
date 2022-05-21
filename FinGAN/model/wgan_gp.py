"""Wasserstein Generative Adversarial Model with Gradient Penalty to ensure 1-Lipschitz continuous Discriminator"""

# Main libs
from typing import Callable, Dict
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns


# In-house modules
from .gan_base import GANBase
from .gan_base import GANmonitoring
from ..utils.parser import Parser


# Set up logger
from DataAnalyst.utils.logging import LoggerConfiguration

log_config = LoggerConfiguration()
log_config.set_logger(log_level="DEBUG", handler="STREAM")
logger = log_config.logger


class WGANGP(tfk.Model, GANBase):

    # Add to keras og init method
    def __init__(
        self,
        Pr: pd.DataFrame,
        dim_latent_space: int = 128,
        D_n: int = 5,
        lambda_: float = 10.0,
    ):
        logger.debug("Initialize tfk.Model")
        super().__init__()
        GANBase.__init__(self, Pr, dim_latent_space)
        self.D_n = D_n
        self.lambda_ = lambda_
        self.G_optimizer = tfk.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )
        self.D_optimizer = tfk.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.5, beta_2=0.9
        )
        self.G_loss = lambda D_Pg: -tf.reduce_mean(D_Pg)  # G tries to maximize D_Pg
        self.D_loss = lambda D_Pr, D_Pg: tf.reduce_mean(D_Pg) - tf.reduce_mean(
            D_Pr
        )  # D tries to minimize D_Pr and maximize D_Pr
        logger.debug("Initialize WGANGP")

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

    # Compute GP
    def _get_GP(self, Pr, Pg):

        if self.D is None or self.G is None:
            raise AttributeError(
                "Make sure to initialize both Discriminator and Generator"
            )

        ## Sidenote: Steps 1-5 are applied on *each* interpolated sample but the computations have been vectorized for efficiency
        # this also mean that batchnormalization isn't allowed anymore within D, as this uses operations on the batch level whereas the GP
        # is computed on the individual sample level. Batchnorm variations can possibly be added if they do not introduce correlations between
        # individual samples.

        # Step 1: Get interpolated sample between Pr and Pg (`x_hat` of Eq.3)
        # x_hat= u*Pr + (1-u)*Pg with u~U(0,1)
        u = tf.random.uniform(shape=(Pr.shape[0], 1), minval=0, maxval=1)
        Prg = u * Pg + (1 - u) * Pr

        # Step 2: Compute Ds output based on the interpolated batch (while taping the computations)
        with tf.GradientTape() as g_tape:
            # Watch the interpolation so we can compute the gradient wrt the interpolated images (and not Ds weights!)
            # https://www.tensorflow.org/api_docs/python/tf/GradientTape
            g_tape.watch(Prg)
            # Forward propogate the interpolation through D
            D_pred_Prg = self.D(
                Prg, training=True
            )  # equivalent but faster than D.predict(Prg) unless multiple batches are used

        # Step 3: Compute the gradients of D with respect to the interpolated batch
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

        if self.D is None or self.G is None:
            raise AttributeError(
                "Make sure to initialize both Discriminator and Generator"
            )

        if isinstance(Pr, tuple):
            Pr = Pr[0]

        if isinstance(
            Pr, pd.DataFrame
        ):  # TODO: conversion must happen much earlier -> create getter for Pr ?
            logger.debug("Converting Pr to numpy ndarray")
            Pr = Pr.to_numpy().astype("float32")

        # Step 1: Train D until convergence / pre-determined number of steps
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
        with tf.GradientTape() as g_tape:
            # Step 2.1: Generate artificial samples from G
            Pg = self.generate_data("Pg", n=Pr.shape[0], as_df=False, to_numpy=False)
            # Step 2.2: Forward propagate samples into D
            D_pred_Pg = self.D(Pg, training=True)
            # Step 2.3: Compute Gs loss
            G_loss = self.G_loss(D_pred_Pg)

        # Step 2.4: Backpropagate loss through D all the way up to Gs weights -> dG_w/dw
        dG_dw = g_tape.gradient(G_loss, self.G.trainable_variables)

        # Step 2.5: Update Gs weights
        self.G_optimizer.apply_gradients(zip(dG_dw, self.G.trainable_variables))

        # Losses
        D_pred_Pr, D_pred_Pg = self.D(Pr), self.D(Pg)
        current_losses = {}
        current_losses["Wasserstein distance"] = np.abs(
            self.D_loss(D_pred_Pr, D_pred_Pg)
        )
        current_losses["D_loss_Pr"] = tf.reduce_mean(D_pred_Pr)
        current_losses["D_loss_Pg"] = tf.reduce_mean(D_pred_Pg)
        current_losses["Gradient penalty"] = self._get_GP(Pr, Pg)
        return current_losses


class WGANGPmonitoring(GANmonitoring):
    def __init__(self, theoretical_mapping: Callable = None, plot_step=100):
        super().__init__(theoretical_mapping=theoretical_mapping, plot_step=plot_step)

    def on_train_begin(self, logs: Dict = None):

        # Create dictionary to store losses
        self.losses = {}
        self.losses["Wasserstein distance"] = []
        self.losses["D_loss_Pr"] = []
        self.losses["D_loss_Pg"] = []
        self.losses["Gradient penalty"] = []

    def on_epoch_end(
        self,
        epoch,
        logs: Dict = None,
    ):

        # Update losses
        for k, v in logs.items():
            self.losses[k].append(v)

        # Plot current progress (on JN)
        if (epoch + 1) % self.plot_step == 0:

            # Plot 6 graphs yielding information on the training progress
            fig, ax = plt.subplots(2, 3, figsize=[10, 7])
            clear_output(wait=True)
            fig.suptitle(f"\n      epoch {epoch+1:05d}", fontsize=10)

            # Plot 1: Wasserstein distance & GP
            for k, v in self.losses.items():
                if k in ["Wasserstein distance", "Gradient penalty"]:
                    ax[0, 0].plot(np.arange(1, epoch + 2), v, label=k)
            ax[0, 0].legend(loc=2)

            # Plot 2: D's losses
            for k, v in self.losses.items():
                if k in ["D_loss_Pr", "D_loss_Pg"]:
                    ax[0, 1].plot(np.arange(1, epoch + 2), v, label=k)
            ax[0, 1].legend(loc=2)

            # Plot 3: Generated PDF vs. real PDF
            Pr = self.model.generate_data(distribution="Pr", n=1000)
            Pg = self.model.generate_data(distribution="Pg", n=1000)
            df = Parser.to_one_df(data=[Pr, Pg])
            if Pr.shape[1] == 1:  # univariate distribution

                sns.kdeplot(
                    data=df,
                    x=df.iloc[:, 0],
                    hue="Dataset",
                    alpha=0.6,
                    ax=ax[0, 2],
                    shade=True,
                )

            if Pr.shape[1] == 2:  # bivariate distribution
                sns.scatterplot(
                    data=df,
                    x=df.iloc[:, 0],
                    y=df.iloc[:, 1],
                    hue="Dataset",
                    style="Dataset",
                    alpha=0.6,
                    ax=ax[0, 2],
                )

            if Pr.shape[1] > 2:  # TODO: map to latent space
                for i in range(df.shape[1] - 1):

                    sns.kdeplot(
                        data=df,
                        x=df.iloc[:, i],
                        hue="Dataset",
                        alpha=0.6,
                        ax=ax[0, 2],
                        shade=True,
                    )
            ax[0, 2].legend(loc=2)

            # Plot 4: Generated CDF vs. real CDF
            if Pr.shape[1] == 1:  # Univariate distribution

                sns.ecdfplot(
                    data=df,
                    x=df.iloc[:, 0],
                    hue="Dataset",
                    ax=ax[1, 2],
                )

            if Pr.shape[1] > 1:  # multivariate distribution

                for i in range(df.shape[1] - 1):
                    sns.ecdfplot(
                        data=df,
                        x=df.iloc[:, i],
                        hue="Dataset",
                        ax=ax[1, 2],
                    )
            ax[1, 2].legend(["Pr", "Pg"], loc=2)

            # Plot 5: mapping f: Z -> X vs. G: Z -> X
            if self.model.Pz_distribution.lower() in ["gaussian", "normal"]:
                latent_grid = np.linspace(
                    -3, 3, 100 * self.model.dim_latent_space
                ).reshape(-1, self.model.dim_latent_space)
            elif self.model.Pz_distribution.lower() in ["uniform", "rand"]:
                latent_grid = np.linspace(
                    -1, 1, 100 * self.model.dim_latent_space
                ).reshape(-1, self.model.dim_latent_space)

            y_pred = self.model.G.predict(latent_grid)
            # TODO: multidimension representation for both input/output
            ax[1, 0].plot(latent_grid[:, 0], y_pred[:, 0], label="Generator mapping")
            if self.theoretical_mapping is not None:
                y_true = self.theoretical_mapping(latent_grid[:, 0])
                ax[1, 0].plot(latent_grid[:, 0], y_true, label="Theoretical mapping")
            ax[1, 0].legend(
                loc=2,
            )
            ax[1, 0].set_xlabel("CDF")

            # Plot 6: Ds behavior for a specific dimension
            xmins = np.min(self.model.Pr).to_numpy()
            xmaxs = np.max(self.model.Pr).to_numpy()
            sample_space_grid = np.zeros(shape=(1000, self.model.Pr.shape[1]))
            for i in range(self.model.Pr.shape[1]):
                sample_space_grid[:, i] = np.linspace(xmins[i], xmaxs[i], 1000)

            D_pred_Pr = tfk.activations.sigmoid(self.model.D(sample_space_grid)).numpy()
            ax[1, 1].plot(sample_space_grid, D_pred_Pr, color="k")
            for i in range(sample_space_grid.shape[1]):
                ax[1, 1].fill_between(
                    sample_space_grid[:, i].reshape(-1),
                    D_pred_Pr.reshape(-1),
                    alpha=0.6,
                    color="k",
                )
            ax[1, 1].set_xlabel("sample space")
            ax[1, 1].set_ylabel(r"$P_{D}(x = P_r)$")

            plt.tight_layout()
            plt.show()
