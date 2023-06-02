import os
import fnmatch
import tensorflow as tf
from src.utils.CRPS import *

def DRN_load_model(var_name):
    """
    Load all the saved EMOS global models for a specific variable.

    Args:
        var_name (str): The variable name used in the model files.

    Returns:
        list: A list of TensorFlow models.
    """
    path = "/Data/Delong_BA_Data/models/DRN/"
    # Create the file pattern based on the variable name
    file_pattern = f"DRN_{var_name}_lead_time_*_denormed.h5"

    # List all files in the directory
    files = os.listdir(path)

    # Filter the list to only include .h5 files that match the file pattern
    model_files = [file for file in files if fnmatch.fnmatch(file, file_pattern)]
    
    # Sort the file list based on the lead time
    model_files.sort(key=lambda file: int(file.split('_')[4]))

    # Load each model file and store it in a list
    models = [tf.keras.models.load_model(os.path.join(path, file), custom_objects={
                "crps_cost_function": crps_cost_function,
                "crps_cost_function_trunc": crps_cost_function_trunc
            }) for file in model_files]

    return models


def DRN_load_model_t2m():
    '''
    Function to load t2m EMOS global models
    '''
    return DRN_load_model('t2m')


def DRN_load_model_ws10():
    '''
    Function to load ws10 EMOS global models
    '''
    return DRN_load_model('ws10')   
