"""Visualisation of Neural Networks and their outputs"""


# Main libs
# ---------------
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio

# In-house libs
# -------------
from .parser import Parser


# Plotly configs
# ----------------
pio.renderers.default = "notebook+pdf+vscode+jupyterlab"
pio.templates.default = "ggplot2"
GGPLOT_COLORWAY = ["#F8766D", "#A3A500", "#00BF7D", "#00B0F6", "#E76BF3"]
GGPLOT_COLORWAY2 = [
    "#999999",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


class GANvis:
    """Visualisation class for Generative Adverserial Models"""

    config = {"staticPlot": True}

    @staticmethod
    def plot_decision_boundary(X, y, mdl, resolution=0.1):

        X = Parser._atleast2D(X)
        y = Parser._atleast2D(y)
        # Plot data
        if X.shape[1] == 2:
            # plot scatterplot
            Pr, Pg = Parser.parse_dataset(X, y)
            fig = px.scatter(Pr)
            fig.add_trace(px.scatter(Pr).data[0])
        else:
            # plot distributions as histograms
            Pr, Pg = Parser.parse_dataset(X, y)
            fig = px.histogram(
                Pr,
                nbins=int(Pr.shape[0] / 10),
                histnorm="probability density",
                color_discrete_sequence=[GGPLOT_COLORWAY[0]],
                labels={"variable": "Samples"},
            )
            fig.update_traces(selector=dict(name="feature_1"), name="Real dataset")
            fig.add_trace(
                px.histogram(
                    Pg,
                    nbins=int(Pg.shape[0] / 10),
                    histnorm="probability density",
                    color_discrete_sequence=[GGPLOT_COLORWAY[1]],
                ).data[0]
            )
            fig.update_traces(selector=dict(name="feature_1"), name="Generated dataset")

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
            ymin, ymax = 0, 0.5
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
            # reshape output into a grid
            y_pred = y_pred.reshape(xx.shape)

        else:
            # make vector of predictions along x-dimension and copy x times into y-dimension so we get a grid
            y_pred = mdl.predict(xgrid)[:, 1].reshape((1, -1))
            y_pred = np.tile(y_pred, (len(ygrid), 1))
        # Step 4: Plot predictions as contourplot
        fig.add_contour(
            x=xgrid,
            y=ygrid,
            z=y_pred,
            colorscale=[[0, "#154a21"], [1, "#8a0000"]],
            contours_coloring="heatmap",
            colorbar=dict(
                title=None,  # r"Probability of real sample",
                title_side="right",
                x=1.1,
                ticks="inside",
                # dtick=2,
            ),
        )
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
        fig.update_yaxes(visible=True, showticklabels=True, title=None)
        fig.update_xaxes(visible=True, showticklabels=True, title=None)

        return fig

