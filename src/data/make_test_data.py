# Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
import xarray as xr
import h5py

# Helpful
import time
import datetime
import itertools
from itertools import product

# My Methods
import importlib
import CRPS
import EMOS
from CRPS import *
from EMOS import *

# Load all data
dat_2018 = xr.open_dataset(
    "/mnt/sda/Data2/fourcastnet/data/predictions/ensemble_2018.h5"
).isel(phony_dim_5=slice(1, 51))
dat_2019 = xr.open_dataset(
    "/mnt/sda/Data2/fourcastnet/data/predictions/ensemble_2019.h5"
).isel(phony_dim_5=slice(1, 51))
dat_2020 = xr.open_dataset(
    "/mnt/sda/Data2/fourcastnet/data/predictions/ensemble_2020.h5"
).isel(phony_dim_5=slice(1, 51))
dat_2021 = xr.open_dataset(
    "/mnt/sda/Data2/fourcastnet/data/predictions/ensemble_2021.h5"
).isel(phony_dim_5=slice(1, 51))
dat_2022 = xr.open_dataset(
    "/mnt/sda/Data2/fourcastnet/data/predictions/ensemble_2022.h5"
).isel(phony_dim_5=slice(1, 51))


# Define Variable Names
dat_train_all = [dat_2018, dat_2019, dat_2020, dat_2021]
var_names = ["u10", "v10", "t2m", "t850", "z500"]

# Calc Mean and Std and create new test data file
for var in range(len(var_names)):
    # Set up for file
    start_time = time.time()
    path = "/Data/Delong_BA_Data/Mean_ens_std/" + var_names[var] + '_test.h5'
    f = h5py.File(path, "a")
    name_test = var_names[var] + '_test'
    name_truth = var_names[var] + 'test_truth'
    
    x_test_mean = dat_2022.predictions.isel(phony_dim_2=var).mean(dim="phony_dim_5")
    x_test_std = dat_2022.predictions.isel(phony_dim_2=var).std(dim="phony_dim_5")
    
    x_test = xr.concat([x_test_mean, x_test_std], dim="mean_std")
    x_test = x_test.transpose(
    "phony_dim_0", "phony_dim_1", "phony_dim_3", "phony_dim_4", "mean_std"
)

    y_test = dat_2022.ground_truth.isel(phony_dim_2 = var)
    
    n_days, n_lead_times, lat, long, mean_var = x_test.shape
    
    half_time = time.time()
    time_difference_half = half_time - start_time
    hours = int(time_difference_half // 3600)
    minutes = int((time_difference_half % 3600) // 60)
    seconds = int(time_difference_half % 60)
    formatted_time_half = f" Round {var} finished concatenation in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time_half}")
    
    # Create those files
    try:
        test = f.create_dataset(
            name_test,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )
    except:
        del f[name_test]
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
        del f[name_truth]
        truth = f.create_dataset(
            name_truth,
            shape=(n_days, n_lead_times, lat, long),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )
        
        
    # Put Data inside of files
    for i in range(n_days):
        test[i, ...] = x_test.loc[i, ...]
        truth[i, ...] = y_test[i, ...]
        
    end_time = time.time()
    time_difference = end_time - start_time
    hours = int(time_difference // 3600)
    minutes = int((time_difference % 3600) // 60)
    seconds = int(time_difference % 60)
    formatted_time = f" Round {var} finished in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time}")
    f.close()



