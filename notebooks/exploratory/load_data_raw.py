def load_data_raw():
    '''
    Load Raw data and format dimensions
Args:
    None
Returns:
    list: raw datasets from 2018 - 2022
    '''
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
    
    dat_train = [dat_2018, dat_2019, dat_2020, dat_2021]
    
    var_dict = {
    "phony_dim_0": "forecast_date",
    "phony_dim_1": "lead_time",
    "phony_dim_2": "var",
    "phony_dim_3": "lat",
    "phony_dim_4": "lon",
    "phony_dim_5": "ens",
    }
    
    dat_all = []

    for i in range(len(dat_train)):
        dat_all.append(
            dat_train[i]
            .rename_dims(var_dict)
            .transpose("forecast_date", "lead_time", "var", "lat", "lon", "ens")
    )
        
        
    return dat_all
    