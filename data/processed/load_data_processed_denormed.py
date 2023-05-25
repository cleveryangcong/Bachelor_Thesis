import xarray as xr

def load_data_all_train_proc_denorm(chunks=None):
    """
    Load all variable train processed and denormed data and format dimensions
    Args:
        chunks (dict): Chunk sizes to use for the Dask arrays.
    Returns:
        list: processsed dataset of train variables from 2018 - 2021 
              order_var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    """
    # Load all data with chunks
    path = "/Data/Delong_BA_Data/mean_ens_std_denorm/"
    dat_train_u10 = xr.open_dataset(path + "u10_train_denorm.h5")
    dat_train_v10 = xr.open_dataset(path + "v10_train_denorm.h5")
    dat_train_t2m = xr.open_dataset(path + "t2m_train_denorm.h5")
    dat_train_t850 = xr.open_dataset(path + "t850_train_denorm.h5")
    dat_train_z500 = xr.open_dataset(path + "z500_train_denorm.h5")
    dat_train_ws10 = xr.open_dataset(path + "ws10_train_denorm.h5")
    
    dat_train_all = [
        dat_train_u10,
        dat_train_v10,
        dat_train_t2m,
        dat_train_t850,
        dat_train_z500,
        dat_train_ws10
    ]

    var_dict = {
        "phony_dim_0": "forecast_date",
        "phony_dim_1": "lead_time",
        "phony_dim_2": "lat",
        "phony_dim_3": "lon",
        "phony_dim_4": "mean_std",
    }

    dat_all = [dat_train_u10, dat_train_v10, dat_train_t2m, dat_train_t850, dat_train_z500, dat_train_ws10.rename_dims(var_dict)]

    return dat_all

def load_data_u10_train_proc_denorm():
    '''
    load u10 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[0]


def load_data_v10_train_proc_denorm():
    '''
    load v10 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[1]


def load_data_t2m_train_proc_denorm():
    '''
    load t2m train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[2]


def load_data_t850_train_proc_denorm():
    '''
    load t850 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[3]


def load_data_z500_train_proc_denorm():
    '''
    load z500 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[4]

def load_data_ws10_train_proc_denorm():
    '''
    load ws10 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[5]


def load_data_all_test_proc_denorm(chunks=None):
    """
    Load all variable train processed and denormed data and format dimensions
    Args:
        chunks (dict): Chunk sizes to use for the Dask arrays.
    Returns:
        list: processsed dataset of train variables from 2018 - 2021 
              order_var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    """
    # Load all data with chunks
    path = "/Data/Delong_BA_Data/mean_ens_std_denorm/"
    dat_train_u10 = xr.open_dataset(path + "u10_test_denorm.h5")
    dat_train_v10 = xr.open_dataset(path + "v10_test_denorm.h5")
    dat_train_t2m = xr.open_dataset(path + "t2m_test_denorm.h5")
    dat_train_t850 = xr.open_dataset(path + "t850_test_denorm.h5")
    dat_train_z500 = xr.open_dataset(path + "z500_test_denorm.h5")
    dat_train_ws10 = xr.open_dataset(path + "ws10_test_denorm.h5")
    
    dat_train_all = [
        dat_train_u10,
        dat_train_v10,
        dat_train_t2m,
        dat_train_t850,
        dat_train_z500,
        dat_train_ws10
    ]

    var_dict = {
        "phony_dim_0": "forecast_date",
        "phony_dim_1": "lead_time",
        "phony_dim_2": "lat",
        "phony_dim_3": "lon",
        "phony_dim_4": "mean_std",
    }

    dat_all = [dat_train_u10, dat_train_v10, dat_train_t2m, dat_train_t850, dat_train_z500, dat_train_ws10.rename_dims(var_dict)]

    return dat_all

def load_data_u10_test_proc_denorm():
    '''
    load u10 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[0]


def load_data_v10_test_proc_denorm():
    '''
    load v10 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[1]


def load_data_t2m_test_proc_denorm():
    '''
    load t2m test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[2]


def load_data_t850_test_proc_denorm():
    '''
    load t850 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[3]


def load_data_z500_test_proc_denorm():
    '''
    load z500 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[4]

def load_data_ws10_test_proc_denorm():
    '''
    load ws10 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[5]










