"""Vanilla Generative Adversarial Model of Goodfellow (2014)"""

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
from .gan_base import GANBase, GANmonitoring
from ..utils.parser import Parser

# Set up logger
from DataAnalyst.utils.logging import LoggerConfiguration
log_config = LoggerConfiguration()
log_config.set_logger(log_level="DEBUG", handler="STREAM")
logger = log_config.logger


class VGAN(tfk.Model, GANBase):

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

    # Extend Keras' compile method to incorporate multiple optimizers/losses
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
    # The batch size is undetermined and can be set when calling .fit()
    def train_step(self, Pr):

        if self.D is None or self.G is None:
            raise AttributeError(
                "Make sure to initialize both Discriminator and Generator"
            )

        if isinstance(Pr, tuple):
            Pr = Pr[0]

        # Step 1 Train D for a couple of iterations
        for _ in range(self.D_n):

            # Step 1.1 Generate artificial samples from G in current trained state
            Pg = self.generate_data("Pg", n=Pr.shape[0], output_type=tf.Tensor)
            # Step 1.2 Forward propagate samples into D while taping the computations
            with tf.GradientTape() as g_tape:
                D_pred_Pr = self.D(Pr, training=True)
                D_pred_Pg = self.D(Pg, training=True)
                D_pred = tf.concat(values = [D_pred_Pr, D_pred_Pg], axis= 0)
                # Real samples are labelled with 1s and fake zith 0s
                labels = tf.concat([tf.ones((D_pred_Pr.shape[0], 1)), tf.zeros((D_pred_Pg.shape[0], 1))], axis= 0)
                # Add (scaled) noise to labels for more stable training TODO: undo hardcoding [?]
                labels += tf.random.normal(labels.shape) * 0.1
                D_loss = self.D_loss(y_true= labels, y_pred = D_pred)

            # Step 1.3 Backpropagate into D: dD_w/dw -> gradient vector of D wrt to its weights
            dD_dw = g_tape.gradient(D_loss, self.D.trainable_variables)

            # Step 1.4: Update Ds weights
            self.D_optimizer.apply_gradients(zip(dD_dw, self.D.trainable_variables))
        
        # Step 2: Train G based on Ds feedback
        with tf.GradientTape() as g_tape:
            # Step 2.1 Forward propagate samples into D (we only need to propagate generated samples now)
            Pg = self.generate_data("Pg", n=Pr.shape[0], output_type=tf.Tensor)
            D_pred_Pg = self.D(Pg, training=True)
            # Step 2.2 Create labels -> we use the same loss as for D but instead switch up the labels such that we 
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




class VGANmonitoring(GANmonitoring):
    def __init__(self, theoretical_mapping: Callable = None, plot_step=100):
        super().__init__(theoretical_mapping=theoretical_mapping, plot_step=plot_step)
    
        # Create dictionary to store losses
        self.losses = {}
        self.losses["d_loss"] = []
        self.losses["g_loss"] = []
        # Keep track of global epoch
        self.epoch_global = -1


    def on_epoch_end(
        self,
        epoch,
        logs: Dict = None,
    ):

        # Update losses
        for k, v in logs.items():
            self.losses[k].append(v)
        self.epoch_global += 1
        
        # Plot current progress (on JN)
        if (epoch + 1) % self.plot_step == 0:

            # Plot 6 graphs yielding information on the training progress
            fig, ax = plt.subplots(2, 3, figsize=[10, 7])
            clear_output(wait=True)
            fig.suptitle(f"\n      epoch {self.epoch_global+1:05d}", fontsize=10)

            # Plot 1: D and G loss
            for k, v in self.losses.items():
                if k in ["d_loss", "g_loss"]:
                    ax[0, 0].plot(np.arange(1, self.epoch_global + 2), v, label=k)
            ax[0, 0].legend(loc=2)

            # Plot 2: D's losses
            for k, v in self.losses.items():
                if k in ["d_loss", "g_loss"]:
                    ax[0, 1].plot(np.arange(1, self.epoch_global + 2), v, label=k)
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
            ax[1, 2].set_xlabel("CDF")
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
            ax[1, 0].set_xlabel("")

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
