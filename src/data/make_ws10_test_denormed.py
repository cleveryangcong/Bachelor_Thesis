# Basics
import numpy as np
import time

# Data
import xarray as xr
import h5py

# Helpful
from tqdm import tqdm

# Dask
from dask.distributed import Client

# My Methods
import data.raw.load_data_raw as ldr


def denormalize(mean, std, x):
    denormalized = (x * std) + mean
    return denormalized


def main():
    # Start a Dask Client
    client = Client()

    # Define path and file names for the h5 file to be created
    path = "/Data/Delong_BA_Data/mean_ens_std_denorm/ws10_test_denorm.h5"
    f = h5py.File(path, "a")
    name_test = "ws10_test"
    name_truth = "ws10_test_truth"

    # load global means and stds
    means = np.load(
        "/mnt/sda/Data2/fourcastnet/data/stats_v0/global_means.npy"
    ).flatten()[[0, 1, 2, 5, 14]]
    stds = np.load(
        "/mnt/sda/Data2/fourcastnet/data/stats_v0/global_stds.npy"
    ).flatten()[[0, 1, 2, 5, 14]]

    # Load raw data for the years 2018-2021
    dat_raw = ldr.load_data_raw()[4]  # load data for the year
    n_days = dat_raw.predictions.shape[0]  # get number of days

    # Chunk the data using Dask
    dat_raw = dat_raw.chunk({"forecast_date": 1})

    # The rest is largely the same

    if name_test in f:
        del f[name_test]  # delete the dataset if it already exists
    if name_truth in f:
        del f[name_truth]  # delete the dataset if it already exists

    test = f.create_dataset(
        name_test,
        (n_days, 32, 120, 130, 2),
        dtype=np.float32,
        compression="gzip",
        compression_opts=9,
    )
    truth = f.create_dataset(
        name_truth,
        (n_days, 32, 120, 130),
        dtype=np.float32,
        compression="gzip",
        compression_opts=9,
    )

    for forecast_date in tqdm(range(n_days)):
        # These computations will be performed lazily
        u10_year_date_pred = denormalize(
            means[0],
            stds[0],
            dat_raw.predictions.isel(var=0, forecast_date=forecast_date),
        )
        v10_year_date_pred = denormalize(
            means[1],
            stds[1],
            dat_raw.predictions.isel(var=1, forecast_date=forecast_date),
        )
        ws10_pred = np.hypot(u10_year_date_pred, v10_year_date_pred,)

        u10_year_date_truth = denormalize(
            means[0],
            stds[0],
            dat_raw.ground_truth.isel(var=0, forecast_date=forecast_date),
        )
        v10_year_date_truth = denormalize(
            means[1],
            stds[1],
            dat_raw.ground_truth.isel(var=1, forecast_date=forecast_date),
        )

        ws10_tru = np.hypot(u10_year_date_truth, v10_year_date_truth,)

        ws10_pred_mean = ws10_pred.mean(dim="ens")
        ws10_pred_std = ws10_pred.std(dim="ens")

        ws_test = xr.concat([ws10_pred_mean, ws10_pred_std], dim="mean_std")
        ws_test = ws_test.transpose("lead_time", "lat", "lon", "mean_std")

        # When writing to the file, we force computation with 'compute()'
        test[forecast_date, ...] = ws_test.compute()
        truth[forecast_date, ...] = ws10_tru.compute()

    # Close the h5 file
    f.close()

    # Close the Dask client
    client.close()


if __name__ == "__main__":
    main()
