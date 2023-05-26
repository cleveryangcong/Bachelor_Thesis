import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta
import time

def get_ics_data(path, date, output_path):
    control_pl = xr.open_dataset(f'{path}fc_cf_{date}_pl.grb', engine = "cfgrib")
    control_sfc = xr.open_dataset(f'{path}fc_cf_{date}_sfc.grb', engine = "cfgrib")
    pert_pl = xr.open_dataset(f'{path}fc_pf_{date}_pl.grb', engine = "cfgrib")
    pert_sfc = xr.open_dataset(f'{path}fc_pf_{date}_sfc.grb', engine = "cfgrib")
    dataset = xr.Dataset({
    "u10": (["ics", "latitude", "longitude"], np.concatenate([pert_sfc.u10.data, np.expand_dims(control_sfc.u10.data, axis = 0)])),
    "v10": (["ics", "latitude", "longitude"], np.concatenate([pert_sfc.v10.data, np.expand_dims(control_sfc.v10.data, axis = 0)])),
    "t2m": (["ics", "latitude", "longitude"], np.concatenate([pert_sfc.t2m.data, np.expand_dims(control_sfc.t2m.data, axis = 0)])),
    "sp": (["ics", "latitude", "longitude"], np.concatenate([pert_sfc.sp.data, np.expand_dims(control_sfc.sp.data, axis = 0)])),
    "mslp": (["ics", "latitude", "longitude"], np.concatenate([pert_sfc.msl.data, np.expand_dims(control_sfc.msl.data, axis = 0)])),
    "t850": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.t.isel(isobaricInhPa = 1).data, np.expand_dims(control_pl.t.isel(isobaricInhPa = 1).data, axis = 0)])),
    "u1000": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.u.isel(isobaricInhPa = 0).data, np.expand_dims(control_pl.u.isel(isobaricInhPa = 0).data, axis = 0)])),
    "v1000": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.v.isel(isobaricInhPa = 0).data, np.expand_dims(control_pl.v.isel(isobaricInhPa = 0).data, axis = 0)])),
    "z1000": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.z.isel(isobaricInhPa = 0).data, np.expand_dims(control_pl.z.isel(isobaricInhPa = 0).data, axis = 0)])),
    "u850": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.u.isel(isobaricInhPa = 1).data, np.expand_dims(control_pl.u.isel(isobaricInhPa = 1).data, axis = 0)])),
    "v850": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.v.isel(isobaricInhPa = 1).data, np.expand_dims(control_pl.v.isel(isobaricInhPa = 1).data, axis = 0)])),
    "z850": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.z.isel(isobaricInhPa = 1).data, np.expand_dims(control_pl.z.isel(isobaricInhPa = 1).data, axis = 0)])),
    "u500": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.u.isel(isobaricInhPa = 2).data, np.expand_dims(control_pl.u.isel(isobaricInhPa = 2).data, axis = 0)])),
    "v500": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.v.isel(isobaricInhPa = 2).data, np.expand_dims(control_pl.v.isel(isobaricInhPa = 2).data, axis = 0)])),
    "z500": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.z.isel(isobaricInhPa = 2).data, np.expand_dims(control_pl.z.isel(isobaricInhPa = 2).data, axis = 0)])),
    "t500": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.t.isel(isobaricInhPa = 2).data, np.expand_dims(control_pl.t.isel(isobaricInhPa = 2).data, axis = 0)])),
    "z50": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.z.isel(isobaricInhPa = 3).data, np.expand_dims(control_pl.z.isel(isobaricInhPa = 3).data, axis = 0)])),
    "r500": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.r.isel(isobaricInhPa = 2).data, np.expand_dims(control_pl.r.isel(isobaricInhPa = 2).data, axis = 0)])),
    "r850": (["ics", "latitude", "longitude"], np.concatenate([pert_pl.r.isel(isobaricInhPa = 1).data, np.expand_dims(control_pl.r.isel(isobaricInhPa = 1).data, axis = 0)])),
    "tcwv": (["ics", "latitude", "longitude"], np.concatenate([pert_sfc.tcw.data, np.expand_dims(control_sfc.tcw.data, axis = 0)])),
    })
    data_array = dataset.to_stacked_array(new_dim = "var",sample_dims = ["ics", "latitude", "longitude"]).transpose("ics", "var", "latitude", "longitude").reset_index("var")
    data_array = data_array.roll(longitude = -720).isel(latitude = slice(0,720))
    xr.Dataset({"fields" : data_array}).to_netcdf(f"{output_path}{date}.nc")
    print(f"File {date} saved successfully")

if __name__ == '__main__':
    path = "/lsdf/kit/imk-tro/projects/MOD/Gruppe_Grams/nk2448/2023_FourCastNet/"
    output_path = "/pfs/work7/workspace/scratch/vt0186-fcn/data/ecmwf_ics/"
    # Create all dates
    start = datetime.strptime("01-01-2022", "%d-%m-%Y")
    end = datetime.strptime("01-01-2023", "%d-%m-%Y")
    date_list = [start + timedelta(days=x) for x in range(0, (end-start).days)]
    for date in date_list:
        date_string = date.strftime("%Y%m%d_%H")        
        if not os.path.isfile(output_path+date_string+".nc"):
            start_time = time.time()
            get_ics_data(path, date_string, output_path)
            print("--- %s seconds ---" % (time.time() - start_time))
        
