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

def setup(var_num, lead_time):
    '''
    Args: 
        var_num(int): number from 0 - 5
        lead_time (int): number from 0 - 30
    '''
    
    # 1. Load data
    dat_train_denorm = ldpd.load_data_all_train_proc_denorm()
    dat_test_denorm = ldpd.load_data_all_test_proc_denorm()
    
    
    # 2. Split Data
    dat_X_train_lead_all_denorm, dat_y_train_lead_all_denorm = split_var_lead(
    dat_train_denorm
)
    dat_X_test_lead_all_denorm, dat_y_test_lead_all_denorm = split_var_lead(dat_test_denorm)
    
    # 3. Select var and lead_time and change dims
    train_data_mean = dat_X_train_lead_all_denorm[var_num][lead_time].isel(mean_std=0)
    train_data_std = dat_X_train_lead_all_denorm[var_num][lead_time].isel(mean_std=1)
    train_data_y = dat_y_train_lead_all_denorm[var_num][lead_time]
    train_data_mean = np.expand_dims(train_data_mean, axis=-1)
    train_data_std = np.expand_dims(train_data_std, axis=-1)


    test_data_mean = dat_X_test_lead_all_denorm[var_num][lead_time].isel(mean_std=0)
    test_data_std = dat_X_test_lead_all_denorm[var_num][lead_time].isel(mean_std=1)
    test_data_y = dat_y_test_lead_all_denorm[var_num][lead_time]
    test_data_mean = np.expand_dims(test_data_mean, axis=-1)
    test_data_std = np.expand_dims(test_data_std, axis=-1)
    
    
    # 4. pad train and test datasets
    padded_train_data_mean = pad_images(train_data_mean)
    padded_train_data_std = pad_images(train_data_std)
    padded_train_data_y = pad_images_y(train_data_y)
    
    # padded_test_data = pad_images(test_data)
    padded_test_data_mean = pad_images(test_data_mean)
    padded_test_data_std = pad_images(test_data_std)
    padded_test_data_y = pad_images_y(test_data_y)
    
    return padded_train_data_mean, padded_train_data_std, padded_train_data_y, padded_test_data_mean, padded_test_data_std, padded_test_data_y


def main(var_num, lead_time, train_patches = False, learning_rate = 0.01, epochs = 150, batch_size = 64, filters = 16):
    
    padded_train_data_mean, padded_train_data_std, padded_train_data_y, padded_test_data_mean, padded_test_data_std, padded_test_data_y = setup(var_num, lead_time)
    
    # Parameters for Unet class initialization
    v = "tp"  # Or any other value based on your preference

    # Initialize the U-Net model
    unet_model = Unet(v=v, train_patches=train_patches, filters = filters)

    # Build the model with your training data shape
    model = unet_model.build_model(padded_train_data_mean.shape, var_num, learning_rate=learning_rate)
    
    # Create a callback to save the log data into a CSV file after each epoch
    path_log = '/Data/Delong_BA_Data/models/U_net/csv_log/'
    csv_logger = CSVLogger(f"{path_log}_training_log_var_{var_num}_lead_{lead_time}_lr_{learning_rate}_ep_{epochs}_bs_{batch_size}_filters{filters}.csv")
    
    path_model = "/Data/Delong_BA_Data/models/U_net/models/"
    # Create a filename for the model checkpoint using the parameters
    model_filename = f"{path_model}_unet_model_var_{var_num}_lead_{lead_time}_lr_{learning_rate}_ep_{epochs}_bs_{batch_size}_filters{filters}.h5"

    model_checkpoint = ModelCheckpoint(model_filename, save_best_only=True, monitor='val_loss')
    
    
    hist = model.fit(
    [padded_train_data_mean, padded_train_data_std],
    padded_train_data_y,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.25,
    callbacks = [csv_logger, model_checkpoint],
    verbose = 0
)
    
    
if __name__ == "__main__":
    # Call the main function
    
    # Change parameters for different testing
    var_num = 2
    lead_time = 0
    train_patches = False
    learning_rate = 0.01
    epochs = 150
    batch_size = 64
    filters = 16
    
    main(var_num, lead_time, train_patches = train_patches, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, filters= filters)
    
    


    
    
    





