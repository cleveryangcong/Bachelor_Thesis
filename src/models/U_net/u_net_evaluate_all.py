import sys
import argparse

sys.path.append("/home/dchen/BA_CH_EN/")

# Basics
import tensorflow as tf
import xarray as xr
import pandas as pd

# Helpful
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt

# My Methods
from src.utils.CRPS import *  # CRPS metrics
from src.utils.data_split import *  # Splitting data into X and y
from src.utils.drn_make_X_array import *  # Import make train array functions (make_X_array)
from src.models.EMOS import *  # EMOS implementation
from src.models.DRN.DRN_model import *  # DRN implementation
from src.models.EMOS_global.EMOS_global_load_score import *  # Load EMOS_global_scores
from src.models.EMOS_global.EMOS_global_load_model import *  # Load EMOS_global_models
import data.raw.load_data_raw as ldr  # Load raw data
import data.processed.load_data_processed as ldp  # Load processed data normed
import data.processed.load_data_processed_denormed as ldpd  # Load processed data denormed
from src.models.CRPS_baseline.CRPS_load import *  # Load CRPS scores
from src.models.U_net.unet import Unet
from src.models.U_net.u_net_tests_params import *
from src.models.U_net.unet import *
from src.models.U_net.u_net_train import *
from src.models.U_net.u_net_load_preds import *


def unpad_images(images, original_shape=(120, 130)):
    # Calculate padding for height and width
    pad_height = (images.shape[1] - original_shape[0]) // 2
    pad_width = (images.shape[2] - original_shape[1]) // 2

    unpadded_images = images[
        :, pad_height : -pad_height or None, pad_width : -pad_width or None, :
    ]
    return unpadded_images


def unpad_images_y(images, original_shape=(120, 130)):
    # Calculate padding for height and width
    pad_height_t = (images.shape[1] - original_shape[0]) // 2
    pad_height_b = images.shape[1] - pad_height_t - original_shape[0]

    pad_width_l = (images.shape[2] - original_shape[1]) // 2
    pad_width_r = images.shape[2] - pad_width_l - original_shape[1]

    unpadded_images = images[
        :, pad_height_t : -pad_height_b or None, pad_width_l : -pad_width_r or None
    ]
    return unpadded_images

def main(var_num, lead_time):
    # load data
    test_var_mean = []
    test_var_std = []
    test_var_y = []
    for var in range(6):
        (
            padded_test_data_mean,
            padded_test_data_std,
            padded_test_data_y,
        ) = u_net_load_test_data(var, lead_time)
        test_var_mean.append(padded_test_data_mean)
        test_var_std.append(padded_test_data_std)
        test_var_y.append(padded_test_data_y)
    test_target = test_var_y[var_num]
    test_target_unpad = unpad_images_y(test_target) 
    
    predictions_unpad = u_net_load_preds_mean("t2m")[lead_time]
    if var_num in [5]:
        crps = crps_trunc
    else:
        crps = crps_normal
    
    scores= crps(
    mu=predictions_unpad[:, :, :, 0].flatten(),
    sigma=predictions_unpad[:, :, :, 1].flatten(),
    y=test_target_unpad.flatten(),
)
        
    scores = scores.reshape((357, 120, 130)).mean(axis=0)
    path_scores ="/Data/Delong_BA_Data/scores/U_net_5_mean/"
    np.save(f'{path_scores}U_net_mean_var_{var_num}_lead_{lead_time}_scores.npy', scores)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CRPS for a given variable")

    # Add the arguments
    parser.add_argument('var_num', type=int, help='Variable number between 0 and 5')
        
    args = parser.parse_args()
    
    for lead_time in range(31):
        main(args.var_num, lead_time)
        