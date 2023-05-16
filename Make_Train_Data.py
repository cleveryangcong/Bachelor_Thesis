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
var_names = ["t2m"]

# Calculate Mean and Std and create new file 
for var in range(len(var_names)):
    # Set up for file
    start_time = time.time()
    path = "/Data/Delong_BA_Data/Mean_ens_std/" + var_names[var] + '_train.h5'
    f = h5py.File(path, "a")
    name_train = var_names[var] + '_train'
    name_truth = var_names[var] + '_truth'
    
    
    # Concatening the different years
    x_train_mean = xr.concat([dat_2018.predictions.isel(phony_dim_2 = var).mean(dim = "phony_dim_5"),
                             dat_2019.predictions.isel(phony_dim_2 = var).mean(dim = "phony_dim_5"),
                             dat_2020.predictions.isel(phony_dim_2 = var).mean(dim = "phony_dim_5"),
                             dat_2021.predictions.isel(phony_dim_2 = var).mean(dim = "phony_dim_5")],
                             dim = "phony_dim_0")
    x_train_std = xr.concat([dat_2018.predictions.isel(phony_dim_2 = var).std(dim = "phony_dim_5"),
                             dat_2019.predictions.isel(phony_dim_2 = var).std(dim = "phony_dim_5"),
                             dat_2020.predictions.isel(phony_dim_2 = var).std(dim = "phony_dim_5"),
                             dat_2021.predictions.isel(phony_dim_2 = var).std(dim = "phony_dim_5")],
                             dim = "phony_dim_0")
    x_train = xr.concat([x_train_mean, x_train_std], dim="mean_std")
    x_train = x_train.transpose(
    "phony_dim_0", "phony_dim_1", "phony_dim_3", "phony_dim_4", "mean_std"
)

    y_train = xr.concat([dat_2018.ground_truth.isel(phony_dim_2 = var),
                         dat_2019.ground_truth.isel(phony_dim_2 = var),
                         dat_2020.ground_truth.isel(phony_dim_2 = var),
                         dat_2021.ground_truth.isel(phony_dim_2 = var)],
                         dim = "phony_dim_0")
    
    n_days, n_lead_times, lat, long, mean_var = x_train.shape
    
    half_time = time.time()
    time_difference_half = half_time - start_time
    hours = int(time_difference_half // 3600)
    minutes = int((time_difference_half % 3600) // 60)
    seconds = int(time_difference_half % 60)
    formatted_time_half = f" Round {var} finished concatenation in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time_half}")
    
    # Create those files
    try:
        train = f.create_dataset(
            name_train,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
        )
    except:
        del f[name_train]
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
        train[i, ...] = x_train.loc[i, ...]
        truth[i, ...] = y_train[i, ...]
        
    end_time = time.time()
    time_difference = end_time - start_time
    hours = int(time_difference // 3600)
    minutes = int((time_difference % 3600) // 60)
    seconds = int(time_difference % 60)
    formatted_time = f" Round {var} finished in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time}")
    f.close()













