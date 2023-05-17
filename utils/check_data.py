def check_dataset(data, mean, num, p_all=True):
    '''
    Method for checking whether created dataset is correct
    
    input:
        param data: Raw Data with ensemble forecasts used to verify
        param mean: The Data to be checked, mean already taken
        param num: The number of datapoints to be checked
        param p_all: Print all, print only number of correct or all comparisons?
        
    output:
        None
    '''
    
    count = 0
    for i in range(num):

        help1 = np.round(
            data.isel(phony_dim_0=i, phony_dim_1=0, phony_dim_3=0, phony_dim_4=0)
            .mean(dim="phony_dim_5")
            .values,
            5,
        )
        help2 = np.round(
            mean.isel(
                phony_dim_0=i,
                phony_dim_1=0,
                phony_dim_2=0,
                phony_dim_3=0,
                phony_dim_4=0,
            ).values,
            5,
        )
        if p_all:
            print(help1, help2, help1 == help2)
        if help1 == help2:
            count = count + 1
    print(count)