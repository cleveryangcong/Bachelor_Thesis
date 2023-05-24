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
    Function to process and save wind speed prediction and ground truth data.
    """

    # Initialize timer for performance tracking
    start_time = time.time()

    # Define path and file names for the h5 file to be created
    path = "/Data/Delong_BA_Data/Mean_ens_std/ws10_train.h5"
    f = h5py.File(path, "a")
    name_train = "ws10_train"
    name_truth = "ws10_truth"

    # Load raw data for the years 2018-2021
    dat_raw = ldr.load_data_raw()[0:4]

    # Compute the magnitude (absolute value) of wind speed predictions and truths
    ws10_preds = []  # list to hold prediction data
    ws10_truth = []  # list to hold ground truth data
    for year in tqdm(range(4)):
        ws10_preds.append(
            np.hypot(
                dat_raw[year].predictions.isel(var=0),
                dat_raw[year].predictions.isel(var=1),
            )
        )
        ws10_truth.append(
            np.hypot(
                dat_raw[year].ground_truth.isel(var=0),
                dat_raw[year].ground_truth.isel(var=1),
            )
        )

    # Calculate and concatenate mean and standard deviation of wind speed predictions
    ws10_preds_mean = xr.concat(
        [
            ws10_preds[0].mean(dim="ens"),
            ws10_preds[1].mean(dim="ens"),
            ws10_preds[2].mean(dim="ens"),
            ws10_preds[3].mean(dim="ens"),
        ],
        dim="forecast_date",
    )
    ws10_preds_std = xr.concat(
        [
            ws10_preds[0].std(dim="ens"),
            ws10_preds[1].std(dim="ens"),
            ws10_preds[2].std(dim="ens"),
            ws10_preds[3].std(dim="ens"),
        ],
        dim="forecast_date",
    )

    # Concatenate mean and standard deviation data along new 'mean_std' dimension
    ws_train = xr.concat([ws10_preds_mean, ws10_preds_std], dim="mean_std")
    ws_train = ws_train.transpose(
        "forecast_date", "lead_time", "lat", "lon", "mean_std"
    )

    # Concatenate truth data
    y_train = xr.concat(
        [ws10_truth[0], ws10_truth[1], ws10_truth[2], ws10_truth[3]],
        dim="forecast_date",
    )

    # Extract shape of the data
    n_days, n_lead_times, lat, long, mean_var = ws_train.shape

    # Calculate elapsed time and print it
    half_time = time.time()
    time_difference_half = half_time - start_time
    hours = int(time_difference_half // 3600)
    minutes = int((time_difference_half % 3600) // 60)
    seconds = int(time_difference_half % 60)
    formatted_time_half = f" finished concatenation in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time_half}")

    # Create the datasets within the h5 file for 'train' and 'truth' data
    try:
        train = f.create_dataset(
            name_train,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )
    except:
        del f[name_train]  # if dataset already exists, delete it
        train = f.create_dataset(
            name_train,
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
        train[i, ...] = ws_train.loc[i, ...]
        truth[i, ...] = y_train[i, ...]

    # Close the h5 file
    f.close()
    
    
    
if __name__ == "__main__":
    main()