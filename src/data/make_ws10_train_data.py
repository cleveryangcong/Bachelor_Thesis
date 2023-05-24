# Basics
import numpy as np
import time

# Data
import xarray as xr
import h5py

# Helpful
from tqdm import tqdm

# My Methods
import data.raw.load_data_raw as ldr

# My Methods
import data.raw.load_data_raw as ldr

def main():

    # Define path and file names for the h5 file to be created
    path = "/Data/Delong_BA_Data/Mean_ens_std/ws10_train.h5"
    f = h5py.File(path, "a")
    name_train = "ws10_train"
    name_truth = "ws10_truth"
    
    # Getting all n_days shapes of raw_data
    n_days_shape = []
    for year in (range(4)):
        dat_raw = ldr.load_data_raw()[year]
        n_days_shape.append(dat_raw.predictions.shape[0])
    
    # Load raw data for the years 2018-2021
    # process one year at a time
    for year in tqdm(range(4)):
        dat_raw = ldr.load_data_raw()[year]

        # Compute the magnitude (absolute value) of wind speed predictions and truths
        ws10_pred = np.hypot(dat_raw.predictions.isel(var=0), dat_raw.predictions.isel(var=1))
        ws10_tru = np.hypot(dat_raw.ground_truth.isel(var=0), dat_raw.ground_truth.isel(var=1))

        # Calculate mean and standard deviation of wind speed predictions
        ws10_pred_mean = ws10_pred.mean(dim="ens")
        ws10_pred_std = ws10_pred.std(dim="ens")

        # Concatenate mean and standard deviation data along new 'mean_std' dimension
        ws_train = xr.concat([ws10_pred_mean, ws10_pred_std], dim="mean_std")
        ws_train = ws_train.transpose("forecast_date", "lead_time", "lat", "lon", "mean_std")
        
        
        # Create the datasets within the h5 file for 'train' and 'truth' data
        # append data to existing dataset, or create it if it doesn't exist
        try:
            train = f[name_train]
        except:
            train_shape = (sum(n_days_shape), *ws_train.shape[1:])  # adjust dimensions as per your data
            train = f.create_dataset(name_train, shape=train_shape, dtype=np.float32, compression="gzip", compression_opts=9)
        try:
            truth = f[name_truth]
        except:
            truth_shape = (sum(n_days_shape), *ws10_tru.shape[1:])
            truth = f.create_dataset(name_truth, shape=truth_shape, dtype=np.float32, compression="gzip", compression_opts=9)

        # Populate the h5 file with the data
        train[sum(n_days_shape[0:year]):sum(n_days_shape[0:(year + 1)]), ...] = ws_train
        truth[sum(n_days_shape[0:year]):sum(n_days_shape[0:(year + 1)]), ...] = ws10_tru

    # Close the h5 file
    f.close()

    
    
    
if __name__ == "__main__":
    main()