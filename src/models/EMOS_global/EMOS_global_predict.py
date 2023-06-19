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

def EMOS_global_load_model(var_num, lead_time):
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    model_filename = (
        "/Data/Delong_BA_Data/models/EMOS_global/EMOS_glob_"
        + var_names[var_num]
        + "_lead_time_"
        + str(lead_time -1 )
        + "_denormed.h5"
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


def EMOS_global_predict(var_num, lead_time):
    """
    Use the given EMOS_global model to predict on a test dataset.
    Args:
        var_num (integer): number between 0 - 5 for each of the variables["u10", "v10", "t2m", "t850", "z500", "ws10"]
        lead_time (integer): number between 0 - 30 for each lead_time
    """

    # Increment lead_time by one to use in indexing
    lead_time = lead_time + 1
    
    # Load Model
    EMOS_glob = EMOS_global_load_model(var_num, lead_time)

    # Define the variable names
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]

    # Load test data
    test_var_denormed = ldpd.load_data_all_test_proc_denorm()[var_num]

    # Split test data into features and targets
    X_test_var_denormed = test_var_denormed[
        list(test_var_denormed.data_vars.keys())[0]
    ].isel(lead_time=lead_time)

    # Use the EMOS model to make predictions
    EMOS_glob_preds = EMOS_glob.predict(
        [
            X_test_var_denormed.isel(mean_std=0).values.flatten(),
            X_test_var_denormed.isel(mean_std=1).values.flatten(),
        ],
        verbose=0,
    )
    
    # Save the predictions 
    preds_filename = f"/Data/Delong_BA_Data/preds/EMOS_global_preds/EMOS_global_{var_names[var_num]}_lead_{lead_time - 1}_preds.npy"
    np.save(preds_filename, EMOS_glob_preds)

    
def main(var_num, lead_time):
    
    EMOS_global_predict(var_num, lead_time)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Calculate CRPS for a given variable")

    # Add the arguments
    parser.add_argument('var_num', type=int, help='Variable number between 0 and 5')
    # Parse the arguments
    args = parser.parse_args()
    
    # Create a pool of worker processes
    pool = mp.Pool(10)

    # Create a list to store the results
    results = []

    # Call the main function for each lead_time
    for lead_time in [0, 14, 29]:
        result = pool.apply_async(main, args=(args.var_num, lead_time))
        results.append(result)
    
    # Close the pool of worker processes
    pool.close()
    
    # Call get() on each result to raise any exceptions
    for result in results:
        result.get()
    
    # Wait for all processes to finish
    pool.join()

