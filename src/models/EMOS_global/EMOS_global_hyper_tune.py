'''
Hypertune EMOS global for one variable for all lead_times of that variable
'''


# Basics
import numpy as np
import argparse
import multiprocessing as mp

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
    
def EMOS_global_train_hyper(
    var_num,
    lead_time,
    batch_size=4096,
    epochs=10,
    lr=0.001,
    validation_split=0.2,
    optimizer="Adam",
    save = True
):
    """
    Train a global EMOS models for a specific variable and lead_time

    Args: 
        var_num (integer): number between 0 - 5 for each of the variables["u10", "v10", "t2m", "t850", "z500", "ws10"]
        lead_time (integer): number between 0 - 30 for each lead_time
    """

    # Adjust lead_time for 1-based indexing
    lead_time = lead_time + 1

    # Define the names of the variables
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]

    # Load the training data
    train_var_denormed = ldpd.load_data_all_train_proc_denorm()[var_num]

    # Split the data into features and target
    X_train_var_denormed = train_var_denormed[
        list(train_var_denormed.data_vars.keys())[0]
    ].isel(lead_time=lead_time)
    y_train_var_denormed = train_var_denormed[
        list(train_var_denormed.data_vars.keys())[1]
    ].isel(lead_time=lead_time)

    # Define the cost function depending on the variable number
    if var_num in [5]:
        crps = crps_cost_function_trunc
    else:
        crps = crps_cost_function

    # Build and compile the model
    EMOS_glob = build_EMOS_network_keras(
        compile=True, lr=lr, loss=crps, optimizer=optimizer
    )

    

    # Define callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    best_score_callback = BestScoreCallback()
    callbacks = [early_stopping, best_score_callback]
    
    if save:
        model_filename = (
            "/Data/Delong_BA_Data/models/EMOS_global/EMOS_glob_"
            + var_names[var_num]
            + "_lead_time_"
            + str(lead_time - 1)
            + "_denormed.h5"
        )
        model_checkpoint = ModelCheckpoint(
            model_filename, monitor="val_loss", mode="min", save_best_only=True
        )
        callbacks.append(model_checkpoint)
    # Fit the model to the training data
    EMOS_glob.fit(
        [
            X_train_var_denormed.isel(mean_std=0).values.flatten(),
            X_train_var_denormed.isel(mean_std=1).values.flatten(),
        ],
        y_train_var_denormed.values.flatten(),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
    )
    return best_score_callback.get_best_score()


def EMOS_global_hyper_tune(var_num, lead_time, batch_sizes=[4096], epochs=[10], lrs=[0.001], optimizers=["Adam"], validation_split=0.2):

    # Combine the hyperparameters using itertools.product
    combinations = list(product(batch_sizes, epochs, lrs, optimizers))

    # Initialize variables to store the best score and parameters
    best_score = float('inf')
    best_params = None

    # Iterate over all combinations and train your model
    for params in tqdm(combinations):
        # Train your model with the current parameters and obtain a score
        score = EMOS_global_train_hyper(var_num, lead_time, batch_size = params[0], epochs = params[1], lr = params[2], optimizer = params[3], save = False)

        # Check if the current score is better than the previous best score
        if score < best_score:
            best_score = score
            best_params = params
            
    return best_params, best_score


def main():
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    var_num = 2
    lead_time = 0 
    epochs = [10]
    batch_sizes = [1024, 2048, 4096, 8192]
    lrs = [0.1, 0.01, 0.001]
    optimizers = ['Adam', 'SGD']
    best_params, best_score = EMOS_global_hyper_tune(var_num, lead_time, batch_sizes = batch_sizes, epochs = epochs, lrs = lrs, optimizers = optimizers)
    best_parms_score = [best_params, best_score]
    
    
    path = f'/Data/Delong_BA_Data/scores/EMOS_global_hyper_scores/EMOS_global_hyper_{var_names[var_num]}_{lead_time}_{best_score}.pkl'
    with open(path, 'wb') as file:
        pickle.dump(best_parms_score, file)
    
if __name__ == "__main__":
    # Call the main function
    main()