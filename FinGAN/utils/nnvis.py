"""Visualisation of Neural Networks and their outputs"""


# Main libs
# ---------------
from typing import Callable, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import matplotlib.pyplot as plt
import tensorflow.keras as tfk
import seaborn as sns
from met_brewer.palettes import met_brew
from IPython.display import clear_output, display






# In-house libs
# -------------
from .parser import Parser
from ..model.gan_base import GANBase


# Plotly configs
# ----------------
colors= met_brew(name="Juarez", brew_type="discrete")
pio.renderers.default = "notebook+pdf+vscode+jupyterlab"
pio.templates["metbrewer"] = go.layout.Template(
    layout=go.Layout(colorway=colors)
)
pio.templates.default = "plotly_white+presentation+metbrewer"
sns.set_palette(colors)

class GANvis:
    """Visualisation class for Generative Adverserial Models"""

    config = {"staticPlot": True}

    @staticmethod
    def plot():
        x = []
        y = []
        fig, ax = plt.subplots(2,2)
        ax[0,0].set_xlim(0,1)
        ax[0,0].set_ylim(0,1)
        ax[0,1].set_xlim(0,1)
        ax[0,1].set_ylim(0,1)
        
        x.append(np.random.rand())
        y.append(np.random.rand())
        ax[0,1].cla()
        ax[0,0].scatter(x,y)
        ax[0,1].scatter(np.random.rand(100),np.random.rand(100))
        display(plt.gcf())
        clear_output(wait=True)

    @staticmethod
    def plot_snapshot(GAN: GANBase, step: int, theoretical_mapping: Callable=None, resolution: int= 100) -> plt.Axes:


        # Plot 4 graphs yielding information on the training progress
        fig, ax = plt.subplots(2, 2, figsize=[10, 10])
        fig.suptitle(f"\n       {step:05d}", fontsize=10)

        # Plot 1: loss and accuracy graphs
        pass

        # Plot 2: empirical vs. generated distribution
        Pr = GAN.generate_data(distribution="Pr", n=1000)
        Pg = GAN.generate_data(distribution="Pg", n=1000)
        df = Parser.to_one_df(data=[Pr, Pg])
        ax[0,1].cla()
        if Pr.shape[1] == 1:
            
            sns.kdeplot(
                data=df,
                x=df.iloc[:, 0],
                hue="Dataset",
                alpha=0.75,
                ax=ax[0, 1],
                shade=True,
            )
        if Pr.shape[1] == 2:
            sns.scatterplot(
                data=df,
                x=df.iloc[:, 0],
                y= df.iloc[:, 1],
                hue="Dataset",
                style="Dataset",
                alpha=0.75,
                ax=ax[0, 1],
            )
        if Pr.shape[1] > 2: #TODO: map to latent space
            sns.scatterplot(
                data=df,
                x=df.iloc[:, 0],
                y= df.iloc[:, 1],
                hue="Dataset",
                style="Dataset",
                alpha=0.75,
                ax=ax[0, 1],
            )

        # Plot 3: mapping f: Z -> X vs. G: Z -> X
        # TODO: multidimension representation for both input/output
        # Gaussian
        if GAN.Pz_distribution.lower() in ["gaussian", "normal"]:
            latent_grid = np.linspace(-3, 3, resolution*GAN.dim_latent_space).reshape(-1, GAN.dim_latent_space)
        # Uniform
        elif GAN.Pz_distribution.lower() in ["uniform", "rand"]:
            latent_grid = np.linspace(-1, 1, resolution*GAN.dim_latent_space).reshape(-1, GAN.dim_latent_space)
        
        y_pred = GAN.G.predict(latent_grid)
        ax[1,0].cla()
        ax[1,0].plot(latent_grid, y_pred, label= "Generator mapping")
        if theoretical_mapping is not None:
            y_true = theoretical_mapping(latent_grid)
            ax[1,0].plot(latent_grid, y_true, label= "Theoretical mapping")

        

        # Plot 4: Ds behavior for a specific dimension
        pass

        display(plt.gcf())
        clear_output(wait=True)

    @staticmethod
    def plot_decision_boundary(
        data: Union[pd.DataFrame, "tuple[np.ndarray]", "tuple[pd.DataFrame]"],
        mdl,
        resolution=0.1,
        col_names: "list[str]" = None,
    ) -> go.Figure:

        # Parse data as machine learning tuples (X,y)
        if isinstance(data, pd.DataFrame) or isinstance(data[0], pd.DataFrame):
            X, y = Parser.to_ml_tuple(data)
        else:
            X, y = data
            X, y = Parser._atleast2D(X), Parser._atleast2D(y)

        # Plot datasets
        if X.shape[1] == 2:  # scatterplot
            fig = GANvis.plot_scatter(data=(X, y), col_names=col_names)
        else:  # histogram
            fig = GANvis.plot_hist(data=(X, y), col_names=col_names)
            # get max y-value to ensure overlaying of contour is nicely fitted
            full_fig = fig.full_figure_for_development(warn=False)
            ymax = full_fig.layout.yaxis.range[1] + 0.1

        # Create Contour plot of prediction space overlayed on original plot
        # -------------------------------------------------------------------
        # Step 1: Define domain
        xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
        xgrid = np.arange(xmin, xmax, resolution)
        if X.shape[1] == 2:
            ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
            ygrid = np.arange(ymin, ymax, resolution)
        elif (
            X.shape[1] == 1
        ):  # contour plot is going to be uniform in the y-direction with p() densities overlayed
            ymin, ymax = 0, ymax
            ygrid = np.linspace(ymin, ymax, int(1 / resolution))
        else:
            # TODO: extent to 3 dim
            raise AttributeError("Max number of features should be 2")

        # Step 2: Create a meshgrid -> two grids containing the (x,y) coordinates of our plot
        if X.shape[1] == 2:
            xx, yy = np.meshgrid(xgrid, ygrid)
        else:
            pass  # in case univariate model a meshgrid isn't necessary

        # Step 3: Get predictions
        if X.shape[1] == 2:
            # flatten meshgrid into one matrix
            xflat, yflat = xx.flatten(), yy.flatten()
            xflat, yflat = (
                xflat.reshape((len(xflat), -1)),
                yflat.reshape((len(yflat), -1)),
            )
            meshgrid = np.hstack((xflat, yflat))
            # make predictions
            y_pred = mdl.predict(meshgrid)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1].reshape((-1, 1))
            else:
                y_pred = y_pred.reshape((-1, 1))
            # reshape output into a grid
            y_pred = y_pred.reshape(xx.shape)

        else:
            # make vector of predictions along x-dimension and copy x times into y-dimension so we get a grid
            y_pred = mdl.predict(xgrid)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1].reshape(-1, 1)
            else:
                y_pred = y_pred.reshape(-1, 1)
            y_pred = np.tile(y_pred, (len(ygrid), 1))
        # Step 4: Plot predictions as contourplot
        fig_contour = go.Figure(
            data=go.Contour(
                x=xgrid,
                y=ygrid,
                z=y_pred,
                colorscale=[[0, colors[0]], [1, colors[1]]],
                contours_coloring="heatmap",
                colorbar=dict(
                    title=None,  # r"Probability of real sample",
                    title_side="right",
                    x=1.1,
                    ticks="inside",
                    # dtick=2,
                ),
            )
        )
        fig_contour.update_traces(name="contourplot")
        fig.add_trace(fig_contour.data[0])
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.4)",
                font_color="white",
            ),
            title=r"$\text{Decision boundary of the Descriminator:} P(x \in \mathbb{P_r})$",
        )
        if X.shape[1] == 1:
            fig.update_yaxes(visible=True, showticklabels=True, title=None)
            fig.update_xaxes(visible=True, showticklabels=True, title=None)
        fig.update_traces(
            selector=dict(name="contourplot"),
            hovertemplate=r"Sample value: %{x: .2f} <br>D's p() of being a real sample = %{z: ,.0%}",
        )

        return fig

    @staticmethod
    def plot_hist(
        data: Union[pd.DataFrame, "tuple[np.ndarray]", "tuple[pd.DataFrame]"],
        kde: bool = False,
        col: str = None,
        col_names: "list[str]" = None,
    ) -> go.Figure:

        if not isinstance(data, pd.DataFrame):
            if col_names is not None:
                df = Parser.to_one_df(data=data, feature_names=col_names)
            else:
                df = Parser.to_one_df(data)
        else:
            df = data.copy()

        col = df.columns[0] if not col else col

        if kde:
            Pr, Pg = Parser.to_two_dfs(df, feature_names=col_names)
            Pr, Pg = Pr[col].to_numpy(), Pg[col].to_numpy()
            fig = ff.create_distplot(
                hist_data=[Pr.tolist(), Pg.tolist()],
                group_labels=["Pr", "Pg"],
                show_hist=True,
                bin_size=0.1,
            )
        else:
            fig = px.histogram(
                data_frame=df,
                x=col,
                color=df["Dataset"],
                opacity=0.5,
                nbins=int(len(df) / 10),
                histnorm="probability density",
                labels={"Dataset": "", "value": col},
            )

        fig.update_yaxes(visible=True, showticklabels=True, title=None)
        fig.update_layout(title=r"$\text{Distributions of } Pr \text{ and } Pg$")
        return fig

    @staticmethod
    def plot_scatter(
        data: Union[pd.DataFrame, "tuple[np.ndarray]", "tuple[pd.DataFrame]"],
        cols: "list[str]" = None,
        col_names: "list[str]" = None,
    ) -> go.Figure:

        if not isinstance(data, pd.DataFrame):
            if col_names is not None:
                df = Parser.to_one_df(data=data, feature_names=col_names)
            else:
                df = Parser.to_one_df(data)
        else:
            df = data.copy()

        if cols is None:
            cols = df.columns[:2]

        fig = px.scatter(
            df,
            x=cols[0],
            y=cols[1],
            color="Dataset",
            marginal_x="histogram",
            marginal_y="histogram",
            labels={"Dataset": ""},
        )
        fig.update_layout(title=r"$\text{Distributions of } Pr \text{ and } Pg$")
        return fig
