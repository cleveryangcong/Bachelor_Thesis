# Helpful
import time
import datetime
import sys

# My Methods
from src.utils.data_split import *
from src.models.EMOS import *
import data.processed.load_data_processed as ldp

def EMOS_global_load_models():
    """
    Load var_lead_time EMOS Models 5x31
Args:
    None
Returns:
    nested list: 5x31 list of trained EMOS Models with one model for each variabel and lead_time
    """
    var_names = ["u10", "v10", "t2m", "t850", "z500"]
    EMOS_global_var_lead_models = [[], [], [], [], []]
    for var in range(5):
        for lead_time in range(31):
            if var in [4]:
                loss_dict = {'crps_cost_function_trunc': crps_cost_function_trunc}
            else:
                loss_dict = {'crps_cost_function': crps_cost_function}
            EMOS_global_var_lead_models[var].append(
                tf.keras.models.load_model(
                    "/home/dchen/BA_CH_EN/models/EMOS_global_models/normed/EMOS_glob_"
                    + var_names[var]
                    + "_lead_time_"
                    + str(lead_time)
                    + "_normed.h5",
                    custom_objects= loss_dict,
                )
            )
    return EMOS_global_var_lead_models