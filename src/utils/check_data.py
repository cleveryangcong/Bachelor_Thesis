import numpy as np
def check_dataset(data, mean, num, p_all=True):
    '''
    Method for checking whether created dataset is correct
    
    input:
        param data: Raw Data with ensemble forecasts used to verify
        param mean: The Data to be checked
        param num: The number of datapoints to be checked
        param p_all: Print all, print only number of correct or all comparisons?
        
    output:
        None
    '''
    
    count = 0
    for i in range(num):

        help1 = np.round(
            data.isel(forecast_date=i, lead_time=0, lat=0, lon=0)
            .mean(dim="ens")
            .values,
            4,
        )
        help2 = np.round(
            mean.isel(
                forecast_date=i,
                lead_time=0,
                lat=0,
                lon=0,
                mean_std = 0
            ).values,
            4,
        )
        if p_all:
            print(help1, help2, help1 == help2)
        if help1 == help2:
            count = count + 1
    print(count)