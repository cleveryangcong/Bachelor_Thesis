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
    """
    Function to process and save wind speed prediction and ground truth data for test dataset.
    """

    # Initialize timer for performance tracking
    start_time = time.time()

    # Define path and file names for the h5 file to be created
    path = "/Data/Delong_BA_Data/Mean_ens_std/ws10_test.h5"
    f = h5py.File(path, "a")
    name_test = "ws10_test"
    name_truth = "ws10_test_truth"

    # Load raw data for the year 2022
    dat_raw = ldr.load_data_raw()[4]

    # Compute the magnitude (absolute value) of wind speed predictions and truths
    ws10_preds = np.hypot(dat_raw.predictions.isel(var=0), dat_raw.predictions.isel(var=1))
    ws10_truth = np.hypot(dat_raw.ground_truth.isel(var=0), dat_raw.ground_truth.isel(var=1))

    # Calculate mean and standard deviation of wind speed predictions
    ws10_preds_mean = ws10_preds.mean(dim="ens")
    ws10_preds_std = ws10_preds.std(dim="ens")

    # Concatenate mean and standard deviation data along new 'mean_std' dimension
    ws_test = xr.concat([ws10_preds_mean, ws10_preds_std], dim="mean_std")
    ws_test = ws_test.transpose("forecast_date", "lead_time", "lat", "lon", "mean_std")

    # Truth data
    y_test = ws10_truth

    # Extract shape of the data
    n_days, n_lead_times, lat, long, mean_var = ws_test.shape

    # Calculate elapsed time and print it
    half_time = time.time()
    time_difference_half = half_time - start_time
    hours = int(time_difference_half // 3600)
    minutes = int((time_difference_half % 3600) // 60)
    seconds = int(time_difference_half % 60)
    formatted_time_half = f" finished concatenation in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time_half}")

    # Create the datasets within the h5 file for 'test' and 'truth' data
    try:
        test = f.create_dataset(
            name_test,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )
    except:
        del f[name_test]  # if dataset already exists, delete it
        test = f.create_dataset(
            name_test,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )
    try:
        truth = f.create_dataset(
            name_truth,
            shape=(n_days, n_lead_times, lat, long),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )
    except:
        del f[name_truth]  # if dataset already exists, delete it
        truth = f.create_dataset(
            name_truth,
            shape=(n_days, n_lead_times, lat, long),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )

    # Populate the h5 file with the data
    for i in range(n_days):
        test[i, ...] = ws_test[i, ...]
        truth[i, ...] = y_test[i, ...]

    # Close the h5 file
    f.close()
    
if __name__ == "__main__":
    main()
