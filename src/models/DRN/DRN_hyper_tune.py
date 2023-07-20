# Basics
import numpy as np
import argparse
import multiprocessing as mp
from itertools import product
import pickle
import pandas as pd

# Tensorflow and Keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

# Xarray
import xarray as xr

# Helpful
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt

# My Methods
from src.utils.CRPS import *  # CRPS metrics
from src.utils.data_split import *  # Splitting data into X and y
from src.utils.drn_make_X_array import *  # Import make train array functions (make_X_array)
from src.models.EMOS import *  # EMOS implementation
from src.models.DRN.DRN_model import *  # DRN implementation
from src.models.EMOS_global.EMOS_global_load_score import *  # Load EMOS_global_scores
from src.models.EMOS_global.EMOS_global_load_model import *  # Load EMOS_global_models
import data.raw.load_data_raw as ldr  # Load raw data
import data.processed.load_data_processed as ldp  # Load processed data normed
import data.processed.load_data_processed_denormed as ldpd  # Load processed data denormed
from src.models.CRPS_baseline.CRPS_load import *  # Load CRPS scores

class BestScoreCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_score = float("inf")  # Initialize the best score as infinity

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        if np.less(current_val_loss, self.best_score):
            self.best_score = current_val_loss  # Update the best score
            
    def get_best_score(self):
        return self.best_score

def DRN_train_hyper(
    var_num,
    lead_time,
    hidden_layer=[],
    emb_size=3,
    max_id=15599,
    batch_size=8192,
    epochs=10,
    lr=0.01,
    validation_split=0.2,
    optimizer="Adam",
    activation="relu",
    save=True,
):
    """
    Trains a Distributional Regression Network (DRN) for weather forecasting.
    The function loads training data, splits it into features and targets, preprocesses the data, 
    builds the model, and trains it. Optionally, the trained model can be saved.

    Args:
        var_num (int): Variable number btw 0-5 to select the variable to be used.
        lead_time (int): The lead time btw 0 - 31. 
        hidden_layer (list): Configurations for the hidden layers.
        emb_size (int): Size of the embedding. Default is 3.
        max_id (int): Maximum ID for the embeddings. Default is 15599. Probably could hard code it inside.
        batch_size (int): Batch size for training. Default is 8192.
        epochs (int): Number of epochs for training. Default is 10.
        lr (float): Learning rate. Default is 0.01.
        validation_split (float): Ratio for validation split. Default is 0.2.
        optimizer (str): Optimizer for training. Default is "Adam".
        activation (str): Activation function. Default is "relu".
        save (bool): Flag to decide whether to save the model or not. Default is True.

    Returns:
        None
    """
    # Define the cost function depending on the variable number
    crps = crps_cost_function_trunc if var_num in [5] else crps_cost_function

    # Define the names of the variables
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]

    # Load all training data of each variable
    train_var_denormed, val_var_denormed = (
        ldpd.load_data_all_train_val_proc_denorm()
    )
    
    # Split the loaded data into features (X) and target (y)
    # also adjusts for lead_time
    dat_X_train_lead_all_denorm, dat_y_train_lead_all_denorm = split_var_lead(
        train_var_denormed
    )
    
    # Split the loaded data into features (X) and target (y)
    # also adjusts for lead_time
    dat_X_val_lead_all_denorm, dat_y_val_lead_all_denorm = split_var_lead(
        val_var_denormed
    )

    # Preprocess the features for Neural Network and scale them
    drn_X_train_lead_array, drn_embedding_train_lead_array = make_X_array(
        dat_X_train_lead_all_denorm, lead_time
    ) 
    
    # Preprocess the features for Neural Network and scale them
    drn_X_val_lead_array, drn_embedding_val_lead_array = make_X_array(
        dat_X_val_lead_all_denorm, lead_time
    ) 

    # Reshape target values into a 1D array
    t2m_y_train = dat_y_train_lead_all_denorm[var_num][lead_time].values.flatten()
    
    
    # Reshape target values into a 1D array
    t2m_y_val = dat_y_val_lead_all_denorm[var_num][lead_time].values.flatten()


    # Build the DRN model with embedding
    drn_lead_model = build_emb_model(
        12,
        2,
        hidden_layer,
        emb_size,
        max_id,
        compile=True,
        optimizer=optimizer,
        lr=lr,
        loss=crps,
        activation=activation,
    )

    # Define callbacks for early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2
    )
    best_score_callback = BestScoreCallback()
    callbacks = [early_stopping, best_score_callback]

    # Check if model saving is requested
    if save:
        # Define the model file path
        model_filename = f"/Data/Delong_BA_Data/models/DRN/DRN_{var_names[var_num]}_lead_time_{lead_time}_denormed.h5"

        # Create a model checkpoint callback to save the model with the minimum validation loss
        model_checkpoint = ModelCheckpoint(
            model_filename, monitor="val_loss", mode="min", save_best_only=True
        )

        # Add the checkpoint callback to the list of callbacks
        callbacks.append(model_checkpoint)

    # Train the DRN model with the prepared data and callbacks
    drn_lead_model.fit(
        [drn_X_train_lead_array, drn_embedding_train_lead_array],
        t2m_y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data = ([drn_X_val_lead_array, drn_embedding_val_lead_array], t2m_y_val),
        callbacks=callbacks,
        verbose=0,
        shuffle = False
    )
    
    return best_score_callback.get_best_score()

if __name__ == "__main__":
    # Create a pool of worker processes
    pool = mp.Pool(10)

    var_num = 5
    hidden_layers = [[512]]
    emb_size = [5]
    epochs = [70]
    batch_sizes = [8192, 16384, 32768]
    lrs = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    optimizers = ['Adam']
    activation = ['relu']
    run = 2 #Always change this
    
    # Combine the hyperparameters using itertools.product
    combinations = list(product(hidden_layers, emb_size, batch_sizes, epochs, lrs, optimizers, activation))

    # Create a DataFrame to store results
    result_df = pd.DataFrame(columns=['lead_time', 'hidden_layer', 'emb_size', 'batch_size', 'epochs', 'lr', 'optimizer', 'activation', 'score'])

    async_results = []
    for lead_time in [0, 15, 30]:
        for params in combinations:
            async_result = pool.apply_async(DRN_train_hyper, args=(var_num, lead_time, params[0], params[1], 15599, params[2], params[3], params[4],0.2,  params[5], params[6], False))
            async_results.append((lead_time, params, async_result))


    # Wait for all processes to finish and collect results
    for lead_time, params, async_result in async_results:
        score = async_result.get()
        result_df = result_df.append({
            'lead_time': lead_time,
            'hidden_layer': params[0], 
            'emb_size': params[1], 
            'batch_size': params[2], 
            'epochs': params[3], 
            'lr': params[4], 
            'optimizer': params[5], 
            'activation': params[6], 
            'score': score}, ignore_index=True)

    # Save the DataFrame to a CSV file
    result_df.to_csv(f'/Data/Delong_BA_Data/scores/DRN_hyper_scores/DRN_hyper_scores_dataframe_{var_num}_run_{run}.csv', index=False)

    # Close the pool of worker processes
    pool.close()
    pool.join()




