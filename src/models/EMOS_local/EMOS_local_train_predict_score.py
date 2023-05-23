













if __name__ == "__main__":

    # Load Dataset:
    dat_train_proc_norm = ldp.load_data_all_train_proc_norm()
    dat_test_proc_norm = ldp.load_data_all_test_proc_norm()
    # Split Dataset into var_lead_lat_lon
    start = time.time()
    X_train_var_lead_lat_lon, y_train_var_lead_lat_lon = split_var_lead_lat_lon(
        dat_train_proc_norm
    )
    print(f"Time after first data split process: {time.time() - start} seconds")
    X_test_var_lead_lat_lon, y_test_var_lead_lat_lon = split_var_lead_lat_lon(
        dat_train_proc_norm
    )
    print(f'Time after second data split process: {time.time() - start} seconds')
    
    # Make pool separate based on variable and lead_time
    for var in range(5):
        for lead_time in range(31):
            # Get datasets for [var][lead_time], with split on 
            EMOS_loc_lat_lon(X_train_var_lead_lat_lon[var][lead_time])
    
    
    main()