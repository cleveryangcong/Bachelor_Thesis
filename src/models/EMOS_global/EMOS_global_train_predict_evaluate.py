# Basics
import numpy as np
import argparse
import multiprocessing as mp

# TensorFlow and Keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

def EMOS_global_train(
    var_num,
    lead_time,
    batch_size=4096,
    epochs=10,
    lr=0.001,
    validation_split=0.2,
    optimizer="Adam",
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

    # Define the filename for the model checkpoint
    model_filename = (
        "/home/dchen/BA_CH_EN/models/EMOS_global_models/denormed/EMOS_glob_"
        + var_names[var_num]
        + "_lead_time_"
        + str(lead_time - 1)
        + "_denormed.h5"
    )

    # Define callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    model_checkpoint = ModelCheckpoint(
        model_filename, monitor="val_loss", mode="min", save_best_only=True
    )

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
        callbacks=[early_stopping, model_checkpoint],
    )

    # Load the best model and return
    best_model = tf.keras.models.load_model(
        model_filename, 
        custom_objects={
            "crps_cost_function": crps_cost_function,
            "crps_cost_function_trunc": crps_cost_function_trunc
        }
    )

    return best_model


def EMOS_global_predict_evaluate(EMOS_glob, var_num, lead_time):
    """
    Use the given EMOS_global model to predict and evaluate on a test dataset.
    Args:
        EMOS_glob: the EMOS global model
        var_num (integer): number between 0 - 5 for each of the variables["u10", "v10", "t2m", "t850", "z500", "ws10"]
        lead_time (integer): number between 0 - 30 for each lead_time
    """

    # Increment lead_time by one to use in indexing
    lead_time = lead_time + 1

    # Define the variable names
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]

    # Load test data
    test_var_denormed = ldpd.load_data_all_test_proc_denorm()[var_num]

    # Split test data into features and targets
    X_test_var_denormed = test_var_denormed[
        list(test_var_denormed.data_vars.keys())[0]
    ].isel(lead_time=lead_time)

    y_test_var_denormed = test_var_denormed[
        list(test_var_denormed.data_vars.keys())[1]
    ].isel(lead_time=lead_time)

    # Choose the CRPS cost function based on variable number
    if var_num in [5]:
        crps = crps_trunc
    else:
        crps = crps_normal

    # Use the EMOS model to make predictions
    EMOS_glob_preds = EMOS_glob.predict(
        [
            X_test_var_denormed.isel(mean_std=0).values.flatten(),
            X_test_var_denormed.isel(mean_std=1).values.flatten(),
        ],
        verbose=1,
    )

    # Evaluate the model predictions using the chosen CRPS function
    EMOS_glob_crps = crps(
        mu=EMOS_glob_preds[:, 0],
        sigma=EMOS_glob_preds[:, 1],
        y=y_test_var_denormed.values.flatten(),
    )
    
    # Reshape the CRPS scores and compute the mean along the first axis
    EMOS_glob_crps = EMOS_glob_crps.reshape(y_test_var_denormed.shape).mean(axis=0)

    # Save the average CRPS score over all days for 120 x 130 grid
    model_filename = f"/Data/Delong_BA_Data/scores/EMOS_global_scores/EMOS_global_{var_names[var_num]}_lead_{lead_time}_scores.npy"
    np.save(model_filename, EMOS_glob_crps)
    
    
    
    
def main(
    var_num, lead_time, batch_size=4096, epochs=10, lr=0.001, validation_split=0.2, optimizer="Adam"
):
    EMOS_glob = EMOS_global_train(
        var_num,
        lead_time,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        validation_split=validation_split,
        optimizer=optimizer
    )
    EMOS_global_predict_evaluate(EMOS_glob, var_num, lead_time)
    

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Calculate CRPS for a given variable")

    # Add the arguments
    parser.add_argument('var_num', type=int, help='Variable number between 0 and 5')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size to use (default: 4096)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--validation_split', type=float, default=0.2, help='validation split(default: 0.2)')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Optimizer to use(default: Adam)')
    # Parse the arguments
    args = parser.parse_args()
    
    # Create a pool of worker processes
    pool = mp.Pool(10)

    # Create a list to store the results
    results = []

    # Call the main function for each lead_time
    for lead_time in range(31):
        result = pool.apply_async(main, args=(args.var_num, lead_time, args.batch_size, args.epochs, args.lr, args.validation_split, args.optimizer))
        results.append(result)
    
    # Close the pool of worker processes
    pool.close()
    
    # Call get() on each result to raise any exceptions
    for result in results:
        result.get()
    
    # Wait for all processes to finish
    pool.join()


