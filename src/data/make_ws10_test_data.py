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


def main():

    # Define path and file names for the h5 file to be created
    path = "/Data/Delong_BA_Data/Mean_ens_std/ws10_test.h5"
    f = h5py.File(path, "a")
    name_test = "ws10_test"
    name_truth = "ws10_test_truth"

    # Load raw data for the year following the train data years
    dat_raw = ldr.load_data_raw()[4]  # Assuming index 4 corresponds to the test year
    n_days = dat_raw.predictions.shape[0]  # get number of days

    # Create the datasets within the h5 file for 'test' and 'truth' data
    # Create them once, before entering the forecast_date loop
    if name_test in f:
        del f[name_test]  # delete the dataset if it already exists
    if name_truth in f:
        del f[name_truth]  # delete the dataset if it already exists

    test = f.create_dataset(name_test, (n_days, *dat_raw.predictions.isel(var=0, forecast_date=0).shape), dtype=np.float32, compression="gzip", compression_opts=9)
    truth = f.create_dataset(name_truth, (n_days, *dat_raw.ground_truth.isel(var=0, forecast_date=0).shape), dtype=np.float32, compression="gzip", compression_opts=9)

    for forecast_date in tqdm(range(n_days)):
        # Compute the magnitude (absolute value) of wind speed predictions and truths
        ws10_pred = np.hypot(dat_raw.predictions.isel(var=0, forecast_date=forecast_date), dat_raw.predictions.isel(var=1, forecast_date=forecast_date))
        ws10_tru = np.hypot(dat_raw.ground_truth.isel(var=0, forecast_date=forecast_date), dat_raw.ground_truth.isel(var=1, forecast_date=forecast_date))

        # Calculate mean and standard deviation of wind speed predictions
        ws10_pred_mean = ws10_pred.mean(dim="ens")
        ws10_pred_std = ws10_pred.std(dim="ens")

        # Concatenate mean and standard deviation data along new 'mean_std' dimension
        ws_test = xr.concat([ws10_pred_mean, ws10_pred_std], dim="mean_std")
        ws_test = ws_test.transpose("lead_time", "lat", "lon", "mean_std")

        # Populate the h5 file with the data
        test[forecast_date, ...] = ws_test
        truth[forecast_date, ...] = ws10_tru

    # Close the h5 file
    f.close()
    
if __name__ == "__main__":
    main()
