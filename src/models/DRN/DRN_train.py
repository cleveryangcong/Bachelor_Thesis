# Basics
import numpy as np
import argparse
import multiprocessing as mp

# Tensorflow and Keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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



def DRN_train(
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
    train_var_denormed = (
        ldpd.load_data_all_train_proc_denorm()
    )
    
    # Split the loaded data into features (X) and target (y)
    # also adjusts for lead_time
    dat_X_train_lead_all_denorm, dat_y_train_lead_all_denorm = split_var_lead(
        train_var_denormed
    )

    # Preprocess the features for Neural Network and scale them
    drn_X_train_lead_array, drn_embedding_train_lead_array = make_X_array(
        dat_X_train_lead_all_denorm, lead_time
    ) 

    # Reshape target values into a 1D array
    t2m_y_train = dat_y_train_lead_all_denorm[var_num][lead_time].values.flatten()

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
        monitor="val_loss", patience=3
    )
    callbacks = [early_stopping]

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
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )
    
def main(
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
    # Run training algorithm
    DRN_train(
        var_num,
        lead_time,
        hidden_layer=hidden_layer,
        emb_size=emb_size,
        max_id=max_id,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        validation_split=validation_split,
        optimizer=optimizer,
        activation=activation,
        save=save,
    )
    
if __name__ == "__main__":
# Create the parser
    parser = argparse.ArgumentParser(description="Calculate CRPS for a given variable (DRN)")

    # Add the arguments
    parser.add_argument('var_num', type=int, help='Variable number between 0 and 5')
    parser.add_argument('--emb_size', type=int, default=3, help='Embedding size (default: 3)')
    # Continue adding the arguments
    parser.add_argument('--max_id', type=int, default=15599, help='Maximum id number (default: 15599)')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function to use (default: relu)')
    parser.add_argument('--save', type=bool, default=True, help='Option to save the model (default: True)')
    parser.add_argument('--hidden_layer', type=str, default="", help='Define hidden layer sizes as comma-separated integers (e.g., "64,128,64"). Default is an empty list.')

    parser.add_argument('--batch_size', type=int, default=4096, help='batch size to use (default: 4096)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--validation_split', type=float, default=0.2, help='validation split(default: 0.2)')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Optimizer to use(default: Adam)')
    # Parse the arguments
    args = parser.parse_args()
    
    # Create a pool of worker processes
    pool = mp.Pool(16)

    # Create a list to store the results
    results = []

    # Call the main function for each lead_time
    for lead_time in range(31):
        result = pool.apply_async(main, args=(args.var_num, lead_time, list(map(int, args.hidden_layer.split(","))), args.emb_size, args.max_id, args.batch_size, args.epochs, args.lr, args.validation_split, args.optimizer, args.activation, args.save))
            results.append(result)

    # Close the pool of worker processes
    pool.close()
    
    # Call get() on each result to raise any exceptions
    for result in results:
        result.get()
    
    # Wait for all processes to finish
    pool.join()
    
    