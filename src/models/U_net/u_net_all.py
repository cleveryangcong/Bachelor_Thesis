import sys

sys.path.append("/home/dchen/BA_CH_EN/")

# Basics
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import Callback
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
from src.models.U_net.u_net_load_data import *


def pad_land_sea_mask(land_sea_mask, desired_shape=(128, 144)):
    # The original land-sea mask has only spatial dimensions.
    # We'll add two extra dimensions at the beginning and the end to match the shape expected by the padding function.
    land_sea_mask = land_sea_mask[np.newaxis, ..., np.newaxis]

    # Apply padding
    padded_mask = pad_images(land_sea_mask, desired_shape)

    # Remove the extra dimensions we added earlier
    padded_mask = np.squeeze(padded_mask)

    return padded_mask

class PrintEveryNCallback(Callback):
    def __init__(self, n):
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n == 0:
            print(f"Epoch {epoch}, train_loss: {logs.get('loss')}, val_loss: {logs.get('val_loss')}")
            
class EarlyStoppingAfterThreshold(EarlyStopping):
    def __init__(self, threshold, **kwargs):
        super(EarlyStoppingAfterThreshold, self).__init__(**kwargs)
        self.threshold = threshold
        self.stopping_condition_met = False

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0)
        if not self.stopping_condition_met and val_loss <= self.threshold:
            self.stopping_condition_met = True
        if self.stopping_condition_met:
            super().on_epoch_end(epoch, logs)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_to_learning_rate, decay_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_to_learning_rate = decay_to_learning_rate
        self.decay_steps = decay_steps

    def __call__(self, step):
        learning_rate_diff = self.initial_learning_rate - self.decay_to_learning_rate
        decayed_learning_rate = learning_rate_diff * (1 - (step / self.decay_steps))
        final_lr = self.initial_learning_rate - decayed_learning_rate

        # keep final_lr at decay_to_learning_rate after decay_steps
        return tf.where(
            tf.less(step, self.decay_steps),
            final_lr,
            self.decay_to_learning_rate,
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_to_learning_rate": self.decay_to_learning_rate,
            "decay_steps": self.decay_steps,
        }
    
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

def main(var_num, lead_time, threshold = 2, train_patches = False,  initial_learning_rate= 0.01, decay_to_learning_rate = 0.01, epochs = 150, batch_size = 64, filters = 16, num_name = 0):
    
    # load land_sea_mask
    land_sea_mask_dummy = np.load(
    "/Data/Delong_BA_Data/land_sea_mask_dummy/land_sea_mask_dummy.npy"
)
    land_sea_mask_dummy = pad_land_sea_mask(land_sea_mask_dummy)
    land_sea_mask_dummy = np.repeat(land_sea_mask_dummy[np.newaxis, ...], 1429, axis=0)
    
    # load data
    train_var_mean = []
    train_var_std = []
    train_var_y = []
    for var in range(6):
        (
            padded_train_data_mean,
            padded_train_data_std,
            padded_train_data_y,
        ) = u_net_load_train_data(var, lead_time)
        train_var_mean.append(padded_train_data_mean)
        train_var_std.append(padded_train_data_std)
        train_var_y.append(padded_train_data_y)
        
        
    # Then, pack all of your input data into a list
    train_inputs = train_var_mean + train_var_std + [land_sea_mask_dummy]

    train_target = train_var_y[var_num]
    
    
    # Parameters for Unet class initialization
    v = "tp"  # Or any other value based on your preference

    # Initialize the U-Net model
    unet_model = Unet(v=v, train_patches=train_patches, filters = filters)
    
    initial_learning_rate = initial_learning_rate
    decay_to_learning_rate = decay_to_learning_rate
    steps_per_epoch = tf.math.ceil(1071 / batch_size)  # round up the result of the division
    decay_steps = int(200 * steps_per_epoch)  # decay over 500 epochs

    lr_schedule = CustomSchedule(initial_learning_rate, decay_to_learning_rate, decay_steps)

    # Build the model with your training data shape
    model = unet_model.build_model(padded_train_data_mean.shape, var_num, learning_rate=lr_schedule)
    
 # Create a callback to save the log data into a CSV file after each epoch
    path_log = '/Data/Delong_BA_Data/models/U_net/csv_log_final/'
    csv_logger = CSVLogger(f"{path_log}training{num_name}_log_var_{var_num}_lead_{lead_time}.csv")
    
    early_stopping = EarlyStoppingAfterThreshold(threshold=threshold, monitor='val_loss', patience=25)
    print_every_n_callback = PrintEveryNCallback(50) # print every 50 epochs

    hist = model.fit(
    train_inputs,
    train_target,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.25,
    callbacks = [csv_logger, early_stopping, print_every_n_callback],  # Removed model_checkpoint from callbacks
    verbose = 0
)
       
    # Prediction part:
    
    # load land_sea_mask
    land_sea_mask_dummy = np.load(
        "/Data/Delong_BA_Data/land_sea_mask_dummy/land_sea_mask_dummy.npy"
    )
    land_sea_mask_dummy = pad_land_sea_mask(land_sea_mask_dummy)
    land_sea_mask_dummy = np.repeat(land_sea_mask_dummy[np.newaxis, ...], 357, axis=0)
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

    # Then, pack all of your input data into a list
    test_inputs = test_var_mean + test_var_std + [land_sea_mask_dummy]

    test_target = test_var_y[var_num]
    
    predictions = model.predict(test_inputs, verbose=0)
        
    predictions_unpad = unpad_images(predictions)
    
    path = '/Data/Delong_BA_Data/preds/U_net_5/'
    
    np.save(f'{path}U_net_{num_name}_var_{var_num}_lead_{lead_time}_preds.npy', predictions_unpad)
        
        
    tf.keras.backend.clear_session()
    
if __name__ == "__main__":
    # Call the main function
    
    # Change parameters for different testing
    var_num = 2
    train_patches = False
    initial_learning_rate = 0.001
    decay_to_learning_rate = 0.0001
    epochs = 3000
    batch_size = 64
    filters = 24
    
    if var_num == 2:
        CRPS_baseline_scores = crps_load_lead_lat_lon("t2m")
        
    elif var_num == 5:
        CRPS_baseline_scores = crps_load_lead_lat_lon("ws10")
    
    for num_name in [1,2]:
        for lead_time in range(31):
            print(f'Begin training lead_time {lead_time}')
            main(var_num, lead_time, threshold = (CRPS_baseline_scores[lead_time].mean()), train_patches = train_patches, initial_learning_rate = initial_learning_rate, decay_to_learning_rate = decay_to_learning_rate, epochs = epochs, batch_size = batch_size, filters= filters, num_name = num_name)
        