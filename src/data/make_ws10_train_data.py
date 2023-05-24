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

    # Load raw data for the years 2018-2021
    # process one year at a time
    for year in tqdm(range(4)):
        dat_raw = ldr.load_data_raw()[year]  # load data for the year
        n_days = dat_raw.predictions.shape[0]  # get number of days

        # Create the datasets within the h5 file for 'train' and 'truth' data
        # Create them once, before entering the forecast_date loop
        if name_train in f:
            del f[name_train]  # delete the dataset if it already exists
        if name_truth in f:
            del f[name_truth]  # delete the dataset if it already exists

        train = f.create_dataset(name_train, (n_days, *dat_raw.predictions.isel(var=0, forecast_date=0).shape), dtype=np.float32, compression="gzip", compression_opts=9)
        truth = f.create_dataset(name_truth, (n_days, *dat_raw.ground_truth.isel(var=0, forecast_date=0).shape), dtype=np.float32, compression="gzip", compression_opts=9)

        for forecast_date in tqdm(range(n_days)):
            # Compute the magnitude (absolute value) of wind speed predictions and truths
            ws10_pred = np.hypot(dat_raw.predictions.isel(var=0, forecast_date=forecast_date), dat_raw.predictions.isel(var=1, forecast_date=forecast_date))
            ws10_tru = np.hypot(dat_raw.ground_truth.isel(var=0, forecast_date=forecast_date), dat_raw.ground_truth.isel(var=1, forecast_date=forecast_date))

            # Calculate mean and standard deviation of wind speed predictions
            ws10_pred_mean = ws10_pred.mean(dim="ens")
            ws10_pred_std = ws10_pred.std(dim="ens")

            # Concatenate mean and standard deviation data along new 'mean_std' dimension
            ws_train = xr.concat([ws10_pred_mean, ws10_pred_std], dim="mean_std")
            ws_train = ws_train.transpose("lead_time", "lat", "lon", "mean_std")

            # Populate the h5 file with the data
            train[forecast_date, ...] = ws_train
            truth[forecast_date, ...] = ws10_tru

    # Close the h5 file
    f.close()

    
if __name__ == "__main__":
    main()