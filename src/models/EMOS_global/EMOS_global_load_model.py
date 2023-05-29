import os
import fnmatch
import tensorflow as tf

def EMOS_global_load_model(var_name):
    """
    Load all the saved EMOS global models for a specific variable.

    Args:
        var_name (str): The variable name used in the model files.

    Returns:
        list: A list of TensorFlow models.
    """
    path = "/home/dchen/BA_CH_EN/models/EMOS_global_models/denormed/"
    # Create the file pattern based on the variable name
    file_pattern = f"EMOS_glob_{var_name}_lead_time_*.h5"

    # List all files in the directory
    files = os.listdir(path)

    # Filter the list to only include .h5 files that match the file pattern
    model_files = [file for file in files if fnmatch.fnmatch(file, file_pattern)]

    # Load each model file and store it in a list
    models = [tf.keras.models.load_model(os.path.join(path, file)) for file in model_files]

    return models

def EMOS_global_load_model_t2m():
    '''
    Function to load t2m EMOS global models
    '''
    return EMOS_global_load_model('t2m')


def EMOS_global_load_model_ws10():
    '''
    Function to load ws10 EMOS global models
    '''
    return EMOS_global_load_model('ws10')   
