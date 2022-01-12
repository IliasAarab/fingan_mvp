"""Wasserstein Generative Adversarial Model with Gradient Penalty to ensure 1-Lipschitz continuous Discriminator"""

# Main libs
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import pandas as pd



class WGANGP(tfk.Model):

    # Add to keras og init method
    def __init__(self, D, G, D_n= 5, lambda_= 10):
        
        super.__init__()
        self.D= D
        self.G= G
        self.D_n= D_n
        self.lambda_= lambda_

    
    # Add to keras og compile method
    def compile(self, D_optimizer, G_optimizer, D_loss, G_loss):

        super.compile()
        self.D_optimizer= D_optimizer
        self.G_optimizer= G_optimizer
        self.D_loss = D_loss
        self.G_loss = G_loss

    # Compute GP 
    def get_GP(self, batch_n, Pr, Pg):

        # Step 1: Get interpolated sample between Pr and Pg (`x_hat` of Eq.3)
        # x_hat= u*Pr +
        alpha= tf.random.normal(shape= (batch_n, 1), mean= 0, stddev= 1)
        diff= Pr - Pg
        Prg= Pr + alpha * diff

        Pr + 0.5 * (Pr - Pg)= (1+N)*Pr - N*Pg




        
