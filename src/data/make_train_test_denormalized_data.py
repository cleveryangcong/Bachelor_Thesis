import xarray as xr
import numpy as np
import multiprocessing as mp
import time

from src.utils.CRPS import *
from src.utils.data_split import *
from src.models.EMOS import *
import data.raw.load_data_raw as ldr
import data.processed.load_data_processed as ldp
from src.models.EMOS_global.EMOS_global_load_models import *


def denormalize(mean, std, x):
    denormalized = (x * std) + mean
    return denormalized


def denormalize_std(std, x):
    denormalized_std = x * std
    return denormalized_std


def main(mean, std, dataset, name):
    """
    Denormalize a Dataset and save it at data directory
    Args:
        mean (float): mean value to denormalize with
        std (float): standard deviation used to denormalize with
        dataset(dataset): xarray dataset with var_train and var_truth dataArrays
        name (String): name to be given to dataset
    Returns:
        None
    """
    denormalized_mean = xr.apply_ufunc(
        denormalize,
        mean,
        std,
        dataset[list(dataset.data_vars.keys())[0]].isel(mean_std=0),
    )
    denormalized_std = xr.apply_ufunc(
        denormalize_std,
        std,
        dataset[list(dataset.data_vars.keys())[0]].isel(mean_std=1),
    )
    denormalized_truth = xr.apply_ufunc(
        denormalize, mean, std, dataset[list(dataset.data_vars.keys())[1]]
    )
    denormalized_train = xr.concat(
        [denormalized_mean, denormalized_std], dim="mean_std"
    )
    denormalized_train = denormalized_train.transpose(
        "forecast_date", "lead_time", "lat", "lon", "mean_std"
    )
    
    denormalized_dataset = xr.Dataset(
        data_vars={list(dataset.data_vars.keys())[0]: denormalized_train, list(dataset.data_vars.keys())[1]: denormalized_truth,}
    )
    denormalized_dataset.to_netcdf(
        "/Data/Delong_BA_Data/mean_ens_std_denorm/" + name + ".h5", format="NETCDF4"
    )
    

if __name__ == "__main__":
    start_time = time.time()
    # Load data with chunks
    dat_train_proc_norm = ldp.load_data_all_train_proc_norm(chunks={"forecast_date": 10})
    dat_test_proc_norm = ldp.load_data_all_test_proc_norm(chunks={"forecast_date": 10})
    var_names = ["u10", "v10", "t2m", "t850", "z500"]S
    means = np.load("/mnt/sda/Data2/fourcastnet/data/stats_v0/global_means.npy").flatten()[[0, 1, 2, 5, 14]]
    stds = np.load("/mnt/sda/Data2/fourcastnet/data/stats_v0/global_stds.npy").flatten()[[0, 1, 2, 5, 14]]
    # Create a pool of worker processes
    pool = mp.Pool(11)
    
    # main(), make denormed datasets
    for var in range(5):
        name = var_names[var] + "_train_denorm"
        pool.apply_async(main, args=(means[var], stds[var], dat_train_proc_norm[var], name))
        
        
    for var in range(5):
        name = var_names[var] + "_test_denorm"
        pool.apply_async(main, args = (means[var], stds[var], dat_test_proc_norm[var], name))
        
        
    pool.close()
    pool.join()

   
    
    
    
    
