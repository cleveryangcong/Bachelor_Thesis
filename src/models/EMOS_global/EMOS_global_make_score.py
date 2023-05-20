# Basics
import numpy as np
    
# Data
import pickle

# Helpful
from tqdm import tqdm
import os

# My Methods
import importlib
from src.utils.CRPS import *
from src.utils.data_split import *
from src.models.EMOS import *
import data.raw.load_data_raw as ldr
import data.processed.load_data_processed as ldp
from src.models.EMOS_global.EMOS_global_load_models import *

def EMOS_global_predict():
    '''
    Make predictinos for EMOS globally trained models with specific variabel and lead_time
Args:
    None
Returns:
    nested_list: 5x31, predictions on test dataset based on models trained for each variable and lead_time
    
    '''
    # 1. Load dataset
    dat_test_proc = ldp.load_data_all_test_proc_norm()
    # 2. Split dataset
    X_test_lead_all, y_test_var_lead_all = split_var_lead(dat_test_proc)
    # 3. Load trained models
    EMOS_global_var_lead_models = EMOS_global_load_models()
    # 4. Predict based on trained models
    EMOS_global_var_lead_preds = [[],[],[],[],[]]
    for var in range(5):
        for lead_time in range(31):
            preds = EMOS_global_var_lead_models[var][lead_time].predict(
            [
                X_test_lead_all[var][lead_time].isel(mean_std=0).values.flatten(),
                X_test_lead_all[var][lead_time].isel(mean_std=1).values.flatten(),
            ],
            verbose=1,
        )
            
            EMOS_global_var_lead_preds[var].append(pred)
    return EMOS_global_var_lead_preds

def main():
    '''
    Save mean scores of EMOS_global
    '''
    # 1. Load dataset
    dat_test_proc = ldp.load_data_all_test_proc_norm()
    # 2. Split dataset
    X_test_lead_all, y_test_var_lead_all = split_var_lead(dat_test_proc)
    # 3. Get predictions
    EMOS_global_var_lead_preds = EMOS_global_predict()
    # 4. Calculate all mean scores
    EMOS_global_scores = crps_var_lead_preds(EMOS_global_var_lead_preds, y_test_var_lead_all)
    EMOS_global_mean_score = [[], [], [], [], []]
    for var in range(5):
        for lead_time in range(31):
            EMOS_global_mean_score[var].append(EMOS_global_scores[var][lead_time].mean())
    # 6. Pickle EMOS Global Mean score
    with open(
        "/Data/Delong_BA_Data/scores/EMOS_global_scores/EMOS_global_mean_scores.pkl",
        "wb",
    ) as f:  # open a text file
        pickle.dump(crps_var_lead_mean_test, f)  # serialize the list
    f.close()

    
    
if __name__ == "__main__":
    # Call the main function
    main()