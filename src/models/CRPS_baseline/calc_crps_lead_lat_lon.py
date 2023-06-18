# Data
from tqdm import tqdm
import numpy as np
import xarray as xr
import argparse

# My Methods
from src.utils.CRPS import *
from src.utils.data_split import *
from src.models.EMOS import *
import data.raw.load_data_raw as ldr
import data.processed.load_data_processed as ldp
import data.processed.load_data_processed_denormed as ldpd

def main(var_num, truncated = True, val = True):
    '''
    Args:
    var_num (integer): number between 0 - 5 for one of the variables
    Return:
    None
    '''
    # Declare which dataset to use
    if val:
        dataframe =  ldpd.load_data_all_train_val_proc_denorm()[1][var_num]
    else:
        dataframe =  ldpd.load_data_all_test_proc_denorm()[var_num]
    
    X_dataframe = dataframe[list(dataframe.data_vars.keys())[0]]
    y_dataframe = dataframe[list(dataframe.data_vars.keys())[1]]
    # Define a list of variable names. var_num is used to select the appropriate variable from this list.
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    var_name = var_names[var_num]
    
    # Check if a truncated CRPS function should be used or a normal one.
    if truncated:
        crps = crps_trunc
    else: 
        crps = crps_normal
    
    # Loop through the lead times from 1 to 31 inclusive.
    for lead_time in tqdm(range(1,32)):
        # Select the corresponding lead time data from the prediction and truth data.
        X_dataframe_lead = X_dataframe.isel(lead_time = lead_time)
        y_dataframe_lead = y_dataframe.isel(lead_time = lead_time)
    
        # Calculate the CRPS values by feeding the mean and std prediction, and the true values into the crps function.
        # The mean is calculated over the forecast_date axis (axis=0).
        crps_values = crps(X_dataframe_lead.isel(mean_std=0).values,
                                  X_dataframe_lead.isel(mean_std=1).values,
                                  y_dataframe_lead.values).mean(axis = 0)
        
        # Save the calculated CRPS values as numpy files.
        if val:
            np.save(f"/Data/Delong_BA_Data/scores/crps_benchmark_scores/{var_name}_lead_{lead_time - 1}_val_scores.npy", crps_values)
        else:
            np.save(f"/Data/Delong_BA_Data/scores/crps_benchmark_scores/{var_name}_lead_{lead_time - 1}_scores.npy", crps_values)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Calculate CRPS for a given variable")

    # Add the arguments
    parser.add_argument('var_num', type=int, help='Variable number between 0 and 5')
    parser.add_argument('--truncated', action='store_true', help='Use truncated CRPS')
    parser.add_argument('--val', action='store_false', help='Use validation set')
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args.var_num, args.truncated, args.val)



