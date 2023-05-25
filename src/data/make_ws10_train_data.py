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

from dask.distributed import Client


def main():
    # Start a Dask Client
    client = Client()

    # Define path and file names for the h5 file to be created
    path = "/Data/Delong_BA_Data/Mean_ens_std/ws10_train.h5"
    f = h5py.File(path, "a")
    name_train = "ws10_train"
    name_truth = "ws10_truth"

    # Calculate total number of days for all years
    total_days = sum(ldr.load_data_raw()[year].predictions.shape[0] for year in range(4))

    # Create datasets with total_days
    if name_train in f:
        del f[name_train]
    if name_truth in f:
        del f[name_truth]

    train = f.create_dataset(
        name_train,
        shape=(total_days, 32, 120, 130, 2),
        dtype=np.float32,
        compression="gzip",
        compression_opts=9,
    )
    truth = f.create_dataset(
        name_truth,
        shape=(total_days, 32, 120, 130),
        dtype=np.float32,
        compression="gzip",
        compression_opts=9,
    )

    start_day = 0
    # Load raw data for the years 2018-2021
    # process one year at a time
    for year in tqdm(range(4)):
        dat_raw = ldr.load_data_raw()[year]  # load data for the year
        n_days = dat_raw.predictions.shape[0]  # get number of days

        # Here we make sure the dataset is chunked for out-of-core computation with dask
        dat_raw = dat_raw.chunk({"forecast_date": 1})

        for forecast_date in tqdm(range(n_days)):
            # These computations will be performed lazily
            ws10_pred = np.hypot(
                dat_raw.predictions.isel(var=0, forecast_date=forecast_date),
                dat_raw.predictions.isel(var=1, forecast_date=forecast_date),
            )
            ws10_tru = np.hypot(
                dat_raw.ground_truth.isel(var=0, forecast_date=forecast_date),
                dat_raw.ground_truth.isel(var=1, forecast_date=forecast_date),
            )

            ws10_pred_mean = ws10_pred.mean(dim="ens")
            ws10_pred_std = ws10_pred.std(dim="ens")

            ws_train = xr.concat([ws10_pred_mean, ws10_pred_std], dim="mean_std")
            ws_train = ws_train.transpose("lead_time", "lat", "lon", "mean_std")

            print("ws_train shape: ", ws_train.shape)
            print("ws10_tru shape: ", ws10_tru.shape)

            # When writing to the file, we force computation with 'compute()'
            train[start_day + forecast_date, ...] = ws_train.compute()
            truth[start_day + forecast_date, ...] = ws10_tru.compute()

        start_day += n_days

    # Close the h5 file
    f.close()
    # Close the Dask client
    client.close()

    
    
if __name__ == "__main__":
    main()