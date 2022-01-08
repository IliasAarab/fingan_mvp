"""Discriminator for a Generative Adversarial Network"""


## Main libs
# ------------------
import tensorflow as tf
import tensorflow.keras as tfk


# Create flexible discriminator initializer
# --------------------------------------------------------------------------------------------------
def init_discriminator(
    number_of_variables,
    number_of_output_units=1,
    output_activation=tfk.activations.linear,
):
    # MVP with hardcoded components
    # using TFs functional API to create a Directed Acylic Graph (DAG)

    # Inputlayer
    dag_input = tfk.Input(shape=(number_of_variables,), name="input_layer")
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
        units=number_of_output_units, activation=output_activation, name="output_layer"
    )(dag)

    # Construct model from DAG
    D = tfk.Model(inputs=dag_input, outputs=dag_output, name="Discriminator")

    # Visualize mdl
    D.summary()
    tfk.utils.plot_model(
        D, to_file="d_mdl.png", show_dtype=True, show_layer_names=True, show_shapes=True
    )

    # Return mdl
    return D


# Create flexible generator initializer
# --------------------------------------------------------------------------------------------------
def init_generator(
    latent_space,
    number_of_variables,
    number_of_output_units=1,
    output_activation=tfk.activations.tanh,
):
    # MVP with hardcoded components
    # using TFs functional API to create a Directed Acylic Graph (DAG)

    # Inputlayer
    dag_input = tfk.Input(shape=(latent_space,), name="input_layer")
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
        units=number_of_variables, activation=output_activation, name="output_layer"
    )(dag)

    # Construct model from DAG
    G = tfk.Model(inputs=dag_input, outputs=dag_output, name="Generator")

    # Visualize mdl
    G.summary()
    tfk.utils.plot_model(
        G, to_file="g_mdl.png", show_dtype=True, show_layer_names=True, show_shapes=True
    )

    # Return mdl
    return G

