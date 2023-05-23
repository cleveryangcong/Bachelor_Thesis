# Data
import xarray as xr

# My Methods
import importlib
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
        data_vars={"u10_train": denormalized_train, "u10_truth": denormalized_truth,}
    )
    denormalized_dataset.to_netcdf(
        "/Data/Delong_BA_Data/mean_ens_std_denorm/" + name + ".h5", format="NETCDF4"
    )
    
if __name__ == "__main__":
    start_time = time.time()
    # Load data
    dat_train_proc_norm = ldp.load_data_all_train_proc_norm()
    dat_test_proc_norm = ldp.load_data_all_test_proc_norm()
    var_names = ["u10", "v10", "t2m", "t850", "z500"]
    
    # Create a pool of worker processes
    pool = mp.Pool(10)
    
    # main(), make denormed datasets
    for var in range(5):
        name = var_names[var] + "_train_denorm"
        pool.apply_async(main(), args = (means[var], stds[var], dat_train_proc_norm[var], name))
        
        
        half_time = time.time()
        time_difference_half = half_time - start_time
        hours = int(time_difference_half // 3600)
        minutes = int((time_difference_half % 3600) // 60)
        seconds = int(time_difference_half % 60)
        formatted_time_half = f" Round train{var} finished concatenation in:{hours} hours, {minutes} minutes, {seconds} seconds"
        print(f"{formatted_time_half}")
        
        
    for var in range(5):
        name = var_names[var] + "_test_denorm"
        pool.apply_async(main(), args = (means[var], stds[var], dat_test_proc_norm[var], name))
        
        
        half_time = time.time()
        time_difference_half = half_time - start_time
        hours = int(time_difference_half // 3600)
        minutes = int((time_difference_half % 3600) // 60)
        seconds = int(time_difference_half % 60)
        formatted_time_half = f" Round test{var} finished concatenation in:{hours} hours, {minutes} minutes, {seconds} seconds"
        print(f"{formatted_time_half}")
        
        
    pool.close()
    pool.join()
    
    
    
    
    