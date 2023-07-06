import sys

sys.path.append("/home/dchen/BA_CH_EN/")

# Basics
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import xarray as xr

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

# Helper functions:    
def pad_images(images, desired_shape=(128, 144)):
    padded_images = np.pad(
        images,
        (
            (0, 0),  # don't pad along the batch axis
            (
                (desired_shape[0] - images.shape[1]) // 2,
                (desired_shape[0] - images.shape[1]) // 2,
            ),  # pad symmetrically along the height
            (
                (desired_shape[1] - images.shape[2]) // 2,
                (desired_shape[1] - images.shape[2]) // 2,
            ),  # pad symmetrically along the width
            (0, 0),
        ),  # don't pad along the channel axis
        mode="reflect",
    )
    return padded_images


def pad_images_y(images, desired_shape=(128, 144)):
    # Calculate padding for height and width
    pad_height = desired_shape[0] - images.shape[1]
    pad_width = desired_shape[1] - images.shape[2]

    # Distribute padding evenly to both sides, add extra to the end if odd difference
    pad_height_t, pad_height_b = pad_height // 2, pad_height - pad_height // 2
    pad_width_l, pad_width_r = pad_width // 2, pad_width - pad_width // 2

    padded_images = np.pad(
        images,
        (
            (0, 0),  # don't pad along the batch axis
            (pad_height_t, pad_height_b),  # pad symmetrically along the height
            (pad_width_l, pad_width_r),  # pad symmetrically along the width
        ),
        mode="reflect",
    )

    return padded_images

def u_net_load_train_data(var_num, lead_time):
    '''
    Args: 
        var_num(int): number from 0 - 5
        lead_time (int): number from 0 - 30
    Returns:
        padded_train_data_mean: Normalized and padded training data (mean)
        padded_train_data_std: Normalized and padded training data (standard deviation)
        padded_train_data_y: Padded y-values for the training data
    '''
    # Load data
    dat_train_denorm = ldpd.load_data_all_train_proc_denorm()

    # Split Data
    dat_X_train_lead_all_denorm, dat_y_train_lead_all_denorm = split_var_lead(dat_train_denorm)
    
    # Select var and lead_time and change dims
    train_data_mean = dat_X_train_lead_all_denorm[var_num][lead_time].isel(mean_std=0)
    train_data_std = dat_X_train_lead_all_denorm[var_num][lead_time].isel(mean_std=1)
    train_data_y = dat_y_train_lead_all_denorm[var_num][lead_time]
    train_data_mean = np.expand_dims(train_data_mean, axis=-1)
    train_data_std = np.expand_dims(train_data_std, axis=-1)
    
    # Normalize Data
    mean_max, std_max = ldpd.load_max_mean_std_values_denorm()
    mean_min, std_min = ldpd.load_min_mean_std_values_denorm()
    train_data_mean_norm = (train_data_mean - mean_min[var_num, lead_time]) / (mean_max[var_num, lead_time] - mean_min[var_num, lead_time])
    train_data_std_norm = (train_data_std - std_min[var_num, lead_time]) / (std_max[var_num, lead_time] - std_min[var_num, lead_time])

    # Pad datasets
    padded_train_data_mean = pad_images(train_data_mean_norm)
    padded_train_data_std = pad_images(train_data_std_norm)
    padded_train_data_y = pad_images_y(train_data_y)
    
    return padded_train_data_mean, padded_train_data_std, padded_train_data_y

def u_net_load_test_data(var_num, lead_time):
    '''
    Args: 
        var_num(int): number from 0 - 5
        lead_time (int): number from 0 - 30
    Returns:
        padded_test_data_mean: Normalized and padded test data (mean)
        padded_test_data_std: Normalized and padded test data (standard deviation)
        padded_test_data_y: Padded y-values for the test data
    '''
    # Load data
    dat_test_denorm = ldpd.load_data_all_test_proc_denorm()

    # Split Data
    dat_X_test_lead_all_denorm, dat_y_test_lead_all_denorm = split_var_lead(dat_test_denorm)

    # Select var and lead_time and change dims
    test_data_mean = dat_X_test_lead_all_denorm[var_num][lead_time].isel(mean_std=0)
    test_data_std = dat_X_test_lead_all_denorm[var_num][lead_time].isel(mean_std=1)
    test_data_y = dat_y_test_lead_all_denorm[var_num][lead_time]
    test_data_mean = np.expand_dims(test_data_mean, axis=-1)
    test_data_std = np.expand_dims(test_data_std, axis=-1)
    
    # Normalize Data
    mean_max, std_max = ldpd.load_max_mean_std_values_denorm()
    mean_min, std_min = ldpd.load_min_mean_std_values_denorm()
    test_data_mean_norm = (test_data_mean - mean_min[var_num, lead_time]) / (mean_max[var_num, lead_time] - mean_min[var_num, lead_time])
    test_data_std_norm = (test_data_std - std_min[var_num, lead_time]) / (std_max[var_num, lead_time] - std_min[var_num, lead_time])

    # Pad datasets
    padded_test_data_mean = pad_images(test_data_mean_norm)
    padded_test_data_std = pad_images(test_data_std_norm)
    padded_test_data_y = pad_images_y(test_data_y)
    
    return padded_test_data_mean, padded_test_data_std, padded_test_data_y
