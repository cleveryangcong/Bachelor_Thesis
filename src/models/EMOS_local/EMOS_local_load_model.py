import os
import fnmatch
import tensorflow as tf
from src.utils.CRPS import *

def EMOS_local_load_model(var_num, lead_time):
    """
    Load all the saved local EMOS models for a specific variable and arrange them in a 2D list.

    Args:
        var_num (int): Variable number between 0 - 5 corresponding to the variables ["u10", "v10", "t2m", "t850", "z500", "ws10"].
        lead_time (int): Lead time number between 0 - 30.

    Returns:
        list: A 2D list (120 x 130) of TensorFlow models.
    """
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    var_name = var_names[var_num]
    path = "/home/dchen/BA_CH_EN/models/EMOS_local_models/denormed/"

    # Create a 2D list for the models
    models = [[None for _ in range(130)] for _ in range(120)]

    # Load each model file and store it in the 2D list
    for lat in range(120):
        for lon in range(130):
            # Create the filename
            filename = f"EMOS_loc_{var_name}_lead_time_{lead_time - 1}_{lat}_{lon}_denormed.h5"
            model_path = os.path.join(path, filename)

            # Load the model and store it in the list
            if os.path.isfile(model_path):
                models[lat][lon] = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        "crps_cost_function": crps_cost_function,
                        "crps_cost_function_trunc": crps_cost_function_trunc,
                    },
                )
    return models
