import xarray as xr

def load_data_all_train_proc_norm():
    """
    Load all variable train processed and normed data and format dimensions
Args:
    None
Returns:
    list: processsed dataset of train variables from 2018 - 2021 
          order_var_names = ["u10", "v10", "t2m", "t850", "z500"]
    
    """
    # Load all data
    dat_train_u10 = xr.open_dataset("/Data/Delong_BA_Data/Mean_ens_std/u10_train.h5")
    dat_train_v10 = xr.open_dataset("/Data/Delong_BA_Data/Mean_ens_std/v10_train.h5")
    dat_train_t2m = xr.open_dataset("/Data/Delong_BA_Data/Mean_ens_std/t2m_train.h5")
    dat_train_t850 = xr.open_dataset("/Data/Delong_BA_Data/Mean_ens_std/t850_train.h5")
    dat_train_z500 = xr.open_dataset("/Data/Delong_BA_Data/Mean_ens_std/z500_train.h5")

    dat_train_all = [
        dat_train_u10,
        dat_train_v10,
        dat_train_t2m,
        dat_train_t850,
        dat_train_z500,
    ]

    var_dict = {
        "phony_dim_0": "forecast_date",
        "phony_dim_1": "lead_time",
        "phony_dim_2": "lat",
        "phony_dim_3": "lon",
        "phony_dim_4": "mean_std",
    }

    dat_all = []

    for i in range(len(dat_train_all)):
        dat_all.append(dat_train_all[i].rename_dims(var_dict))

    return dat_all

def load_data_u10_train_proc_norm():
    '''
    load u10 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[0]


def load_data_v10_train_proc_norm():
        '''
    load v10 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[1]


def load_data_t2m_train_proc_norm():
        '''
    load t2m train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[2]


def load_data_t850_train_proc_norm():
        '''
    load t850 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[3]


def load_data_z500_train_proc_norm():
        '''
    load z500 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[4]