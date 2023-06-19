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


def EMOS_local_load_model_var_lead(var_num, lead_time, lat, lon):
    """
    Load the saved local EMOS model for a specific variable, lead time, latitude, and longitude.

    Args:
        var_num (int): Variable number between 0 - 5 corresponding to the variables ["u10", "v10", "t2m", "t850", "z500", "ws10"].
        lead_time (int): Lead time number between 0 - 30.
        lat (int): Latitude index.
        lon (int): Longitude index.

    Returns:
        TensorFlow model or None if model file does not exist.
    """
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    var_name = var_names[var_num]
    path = "/Data/Delong_BA_Data/models/EMOS_local/"

    # Create the filename
    filename = f"EMOS_loc_{var_name}_lead_time_{lead_time}_{lat}_{lon}_denormed.h5"
    model_path = os.path.join(path, filename)

    # Load the model and return
    if os.path.isfile(model_path):
        return tf.keras.models.load_model(
            model_path,
            custom_objects={
                "crps_cost_function": crps_cost_function,
                "crps_cost_function_trunc": crps_cost_function_trunc,
            },
        )
    return None


def EMOS_local_predict(var_num, lead_time):
    """
    Use stored local EMOS model to predict on a test dataset.

    Args:
        var_num (int): Variable number between 0 - 5 corresponding to the variables ["u10", "v10", "t2m", "t850", "z500", "ws10"].
        lead_time (int): Lead time number between 0 - 30.

    Returns:
        None
    """
    # Adjust lead_time for 1-based indexing
    lead_time += 1

    # Define the variable names
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]

    # Load test data
    test_var_denormed = ldpd.load_data_all_test_proc_denorm()[var_num]

    # Initialize a 3D array to store all predictions for the current lead time
    all_predictions = np.zeros((120, 130, 357, 2))

    for lat in range(120):
        for lon in range(130):
            # Load Models
            model = EMOS_local_load_model_var_lead(var_num, (lead_time - 1), lat, lon)
            
            if model is None:
                continue
            
            # Extract the data for the specific lead_time and grid point
            X_test_var_denormed = test_var_denormed[
                list(test_var_denormed.data_vars.keys())[0]
            ].isel(lead_time=lead_time, lat=lat, lon=lon)

            # Predict using the local model for the grid point
            EMOS_loc_preds = model.predict(
                [
                    X_test_var_denormed.isel(mean_std=0).values.flatten(),
                    X_test_var_denormed.isel(mean_std=1).values.flatten(),
                ],
            )

            # Store the predictions in the 3D array
            all_predictions[lat, lon, :, :] = EMOS_loc_preds

            # Clear session and delete the model
            K.clear_session()
            del model

    # Save all predictions for the current lead time in a single file
    preds_filename = f"/Data/Delong_BA_Data/preds/EMOS_local_preds/EMOS_local_{var_names[var_num]}_lead_{lead_time - 1}_preds.npy"
    np.save(preds_filename, all_predictions)


def main(var_num, lead_time):
    EMOS_local_predict(var_num, lead_time)
    

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Predict values for a given variable")

    # Add the arguments
    parser.add_argument('var_num', type=int, help='Variable number between 0 and 5')
    # Parse the arguments
    args = parser.parse_args()
    
    # Create a pool of worker processes
    pool = mp.Pool(16)

    # Create a list to store the results
    results = []

    # Call the main function for each lead_time
    for lead_time in [0,15,30]:
        result = pool.apply_async(main, args=(args.var_num, lead_time))
        results.append(result)
    
    # Close the pool of worker processes
    pool.close()
    
    # Call get() on each result to raise any exceptions
    for result in results:
        result.get()
    
    # Wait for all processes to finish
    pool.join()