'''
Hypertune EMOS local for one variable for all lead_times of that variable
'''

# Basics
import numpy as np
import argparse
import multiprocessing as mp
from itertools import product
import pickle


# TensorFlow and Keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

# Visualization
import matplotlib.pyplot as plt

# Statistical Functions
from scipy.stats import norm

# Progress Bar
from tqdm import tqdm

# My Methods
from src.utils.CRPS import *
from src.utils.data_split import *
from src.models.EMOS import *
import data.raw.load_data_raw as ldr
import data.processed.load_data_processed as ldp
import data.processed.load_data_processed_denormed as ldpd
from src.models.CRPS_baseline.CRPS_load import *


class BestScoreCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_score = float("inf")  # Initialize the best score as infinity

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        if np.less(current_val_loss, self.best_score):
            self.best_score = current_val_loss  # Update the best score

#     def on_train_end(self, logs=None):
#         print(f"Best validation score = {self.best_score}")
        
    def get_best_score(self):
        return self.best_score

        

def EMOS_local_train_hyper(var_num, lead_time, batch_size=4096, epochs=10, lr=0.001, validation_split=0.2, optimizer="Adam", save=True):
    """
    Train a local EMOS model for a specific variable and lead time for all individual grid points.

    Args: 
        var_num (int): Variable number between 0 - 5 corresponding to the variables ["u10", "v10", "t2m", "t850", "z500", "ws10"].
        lead_time (int): Lead time number between 0 - 30.
        batch_size (int): The number of samples per gradient update for training the model.
        epochs (int): The number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        validation_split (float): The fraction of the training data to be used as validation data.
        optimizer (str): The optimizer to use. Default is "Adam".
        save (bool): If True, saves the trained model.

    Returns:
        None
    """
    # Adjust lead_time for 1-based indexing
    lead_time = lead_time + 1

    # Define the cost function depending on the variable number
    crps = crps_cost_function_trunc if var_num in [5] else crps_cost_function

    # Define the names of the variables
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]

    # Load the training data for gridpoint
    train_var_denormed = ldpd.load_data_all_train_proc_denorm()[var_num]
    
    # Best scores over all lat - lon
    best_scores = []

    # Split the data into features and target
    for lat in range(0,120, 10):
        for lon in range(0, 130, 10):
            # Split the data into features and target
            X_train_var_denormed = train_var_denormed[
                list(train_var_denormed.data_vars.keys())[0]
            ].isel(lead_time=lead_time, lat=lat, lon=lon)
            y_train_var_denormed = train_var_denormed[
                list(train_var_denormed.data_vars.keys())[1]
            ].isel(lead_time=lead_time, lat=lat, lon=lon)

            # Build and compile the model
            EMOS_loc = build_EMOS_network_keras(
                compile=True, lr=lr, loss=crps, optimizer=optimizer
            )

            # Define callbacks for early stopping and best_score_callback
            early_stopping = EarlyStopping(monitor="val_loss", patience=3)
            best_score_callback = BestScoreCallback()
            callbacks = [early_stopping, best_score_callback]
            
            if save:
                # Save the model
                model_filename = f"/Data/Delong_BA_Data/models/EMOS_local/EMOS_loc_{var_names[var_num]}_lead_time_{lead_time - 1}_{lat}_{lon}_denormed.h5"
                model_checkpoint = ModelCheckpoint(
                    model_filename, monitor="val_loss", mode="min", save_best_only=True
                )
                callbacks.append(model_checkpoint)

            # Fit the model to the training data
            EMOS_loc.fit(
                [
                    X_train_var_denormed.isel(mean_std=0).values.flatten(),
                    X_train_var_denormed.isel(mean_std=1).values.flatten(),
                ],
                y_train_var_denormed.values.flatten(),
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            best_scores.append(best_score_callback.get_best_score())
    return np.mean(best_scores)
            
            
            

def EMOS_local_hyper_tune(var_num, lead_time, batch_sizes=[4096], epochs=[10], lrs=[0.001], optimizers=["Adam"], validation_split=0.2):

    # Combine the hyperparameters using itertools.product
    combinations = list(product(batch_sizes, epochs, lrs, optimizers))

    # Initialize variables to store the best score and parameters
    best_score = float('inf') # It should be positive infinity
    best_params = None
    all_scores = []
    all_params = []
    
    for params in tqdm(combinations):
        score = EMOS_local_train_hyper(var_num, lead_time, batch_size = params[0], epochs = params[1], lr = params[2], optimizer = params[3], save = False)
        
        all_scores.append(score)
        all_params.append(params)
        # Check if the current score is better than the previous best score
        if score < best_score: # It should check if the score is lower
            best_score = score
            best_params = params
            

    return best_params, best_score, all_params, all_scores


def main():
    for i in [0, 15, 30]:
        var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
        var_num = 2
        lead_time = i
        epochs = [30]
        batch_sizes = [32, 64, 128, 256]
        lrs = [0.1, 0.01, 0.001]
        optimizers = ['Adam', 'SGD']
        best_params, best_score, all_params, all_scores = EMOS_local_hyper_tune(var_num, lead_time, batch_sizes = batch_sizes, epochs = epochs, lrs = lrs, optimizers = optimizers)
        best_parms_score = [best_params, best_score, lead_time, all_params, all_scores]


        path = f'/Data/Delong_BA_Data/scores/EMOS_local_hyper_scores/EMOS_local_hyper_{var_names[var_num]}_{lead_time}_{best_score}.pkl'
        with open(path, 'wb') as file:
            pickle.dump(best_parms_score, file)
    
if __name__ == "__main__":
    # Call the main function
    main()
    
    
    
    