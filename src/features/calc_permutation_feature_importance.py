import os
import multiprocessing as mp

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "-1"  # this line tells TensorFlow not to use any GPU
# Basics
# Basics
import tensorflow as tf
import xarray as xr
import random

# Helpful
import tqdm

# Visualization
import matplotlib.pyplot as plt

# Path setup
import sys

sys.path.append("/home/dchen/BA_CH_EN/")


# My Methods
from src.utils.CRPS import *  # CRPS metrics
from src.utils.data_split import *  # Splitting data into X and y
from src.utils.drn_make_X_array import *  # Import make train array functions (make_X_array)
from src.models.EMOS import *  # EMOS implementation
from src.models.EMOS_global.EMOS_global_load_score import *  # Load EMOS_global_scores
from src.models.EMOS_global.EMOS_global_load_model import *  # Load EMOS_global_models
from src.models.EMOS_local.EMOS_local_load_score import *  # Load EMOS_local_scores
from src.models.EMOS_local.EMOS_local_load_model import *  # Load EMOS_local_models
from src.models.DRN.DRN_model import *  # DRN implementation
from src.models.DRN.DRN_load_score import *  # Load DRN_scores
from src.models.DRN.DRN_load_score_10 import *  # Load DRN_scores_10
from src.models.DRN.DRN_load_score_dummy_10 import *  # Load DRN_scores_10
import data.raw.load_data_raw as ldr  # Load raw data
import data.processed.load_data_processed as ldp  # Load processed data normed
import data.processed.load_data_processed_denormed as ldpd  # Load processed data denormed
from src.models.CRPS_baseline.CRPS_load import *  # Load CRPS scores
from src.models.U_net.u_net_load_score import *
from src.models.U_net.u_net_load_preds import *



def permutation_feature_importance(var_num, lead_time, variable):
    """
    Calculate feature importance based on permutations feature importance
    Args:
        var_num: 0 - 5
        lead_time: 0 - 31
        variable: 0 - 11 with order (u10_mean, u10_std, v10_mean, v10_std, t2m_mean ... grid_embedding)
    """
    # Define the names of the variables
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]

    path = f"/Data/Delong_BA_Data/models/DRN_10_dummy/DRN_{var_names[var_num]}_lead_time_{lead_time}_0_denormed_dummy.h5"
    DRN_model = tf.keras.models.load_model(
        path,
        custom_objects={
            "crps_cost_function": crps_cost_function,
            "crps_cost_function_trunc": crps_cost_function_trunc,
        },
    )

    # Add dummy variable for land sea mask
    land_sea_mask_dummy = np.load(
        "/Data/Delong_BA_Data/land_sea_mask_dummy/land_sea_mask_dummy.npy"
    )
    land_sea_mask_input = np.tile(land_sea_mask_dummy.flatten(), 357)
    # Load all test data of each variable
    test_var_denormed = ldpd.load_data_all_test_proc_denorm()

    # Split the loaded data into features (X) and target (y)
    dat_X_test_lead_all_denorm, dat_y_test_lead_all_denorm = split_var_lead(
        test_var_denormed
    )

    # Preprocess the features for Neural Network and scale them
    drn_X_test_lead_array, drn_embedding_test_lead_array = make_X_array(
        dat_X_test_lead_all_denorm, lead_time
    )
    drn_X_test_lead_array_mask = np.column_stack(
        (drn_X_test_lead_array, land_sea_mask_input)
    )
    drn_X_test_lead_array_reshaped = drn_X_test_lead_array_mask.reshape(
        357, 120, 130, 13
    )
    drn_embedding_test_lead_array_reshaped = drn_embedding_test_lead_array.reshape(
        357, 120, 130
    )
    # Reshape target values into a 1D array
    t2m_y_test_reshaped = dat_y_test_lead_all_denorm[var_num][lead_time].values

    # Load dummy scores:
    DRN_scores_t2m_dummy_10 = DRN_load_score_dummy_10(var_names[var_num])

    permutation_feature_importance = np.zeros([120, 130])
    for lat in tqdm(range(120)):
        for lon in range(130):
            helper_array = np.copy(drn_X_test_lead_array_reshaped)
            helper_array[:, lat, lon, variable] = np.random.permutation(
                drn_X_test_lead_array_reshaped[:, lat, lon, variable]
            )

            DRN_score_grid_point = DRN_model.evaluate(
                [
                    helper_array[:, lat, lon, :],
                    drn_embedding_test_lead_array_reshaped[:, lat, lon],
                ],
                t2m_y_test_reshaped[:, lat, lon],
                verbose=0,
            )
            permutation_feature_grid = (
                DRN_score_grid_point - DRN_scores_t2m_dummy_10[lead_time][lat, lon]
            )
            permutation_feature_importance[lat, lon] = permutation_feature_grid
    path = "/Data/Delong_BA_Data/scores/permutation_feature_importance/"
    np.save(f"{path}permutation_feature_importance_var_{var_num}_lead_{lead_time}_variable_{variable}.npy", permutation_feature_importance)


if __name__ == "__main__":
    # Call the main function
    
    var_num = 5
    lead_time = 11
    
    # Create a pool of worker processes
    pool = mp.Pool(12)

    # Create a list to store the results
    results = []
    for variable in range(12):
        res = pool.apply_async(permutation_feature_importance, args=(var_num, lead_time, variable))
        results.append(res)
        
    # Close the pool of worker processes. No more tasks will be accepted.
    pool.close()
    
    # Wait for all processes to finish
    pool.join()

    # Call get() on each result to raise any exceptions
    for result in results:
        result.get()







