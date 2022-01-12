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
        
        ## Sidenote: Steps 1-5 are applied on _each_ interpolated sample but the computations have been vectorized for efficiency
        # this also mean that batchnormalization isn't allowed anymore within D, as this uses operations on the batch level whereas the GP
        # is computed on the individual sample level. Batchnorm variations can be possibly be added if they do not introduce correlations between 
        # individual samples.
        
        # Step 1: Get interpolated sample between Pr and Pg (`x_hat` of Eq.3)
        # x_hat= u*Pr + (1-u)*Pg with u~U(0,1)
        u= tf.random.uniform(shape= (batch_n, 1), minval=0, maxval=1)
        Prg= u*Pr + (1-u)*Pg

        # Step 2: Compute Ds output based on the interpolated batch (while taping the computations)
        with tf.GradientTape() g_tape:
            # Watch the interpolation so we can compute gradient wrt interpolated images (and not Ds weights!)
            # https://www.tensorflow.org/api_docs/python/tf/GradientTape
            g_tape.watch(Prg)
            # Forward propogate the interpolation through D
            D_pred_Prg= D(Prg, training= True) #equivalent but faster than D.predict(Prg) unless multiple batches are used
        
        # Step 3: Compute the gradients of Ds weights with respect to the interpolated batch
        # so D_pred_Prg = D(Prg) and we need dD(Prg)/dPrg 
        dD_dPrg= g_tape.gradient(D_pred_Prg, [Prg])
        
        # Step 4: Compute the norm of the gradient vector
        



        
