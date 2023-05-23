# Helpful
import time
import datetime
import sys

# My Methods
from src.utils.data_split import *
from src.utils.CRPS import *
from src.models.EMOS import *
import data.processed.load_data_processed as ldp


def main(batch_size = 5000, epochs = 5, lr = 0.1, validation_split = 0.2):
    '''
    Train 5 * 31 models globally, one for each variable and lead_time
Args:
    batch_size (int): batch_size used for model training
    epochs (int): epochs used for model training
    lr(float): learning rate used for model training
    validation_split(float): validation_split used for model training
Returns:
    None:
    '''
    # Basics
    var_names = ["u10", "v10", "t2m", "t850", "z500"]
    
    # 1. Load Data
    dat_train_proc = ldp.load_data_all_train_proc_norm()
    dat_test_proc = ldp.load_data_all_test_proc_norm()
    
    # 2. Data split
    X_train_var_lead_all, y_train_var_lead_all = split_var_lead(dat_train_proc)
    X_test_lead_all, y_test_var_lead_all = split_var_lead(dat_test_proc)
    
    
    num = 0
    for var in range(5):
        start_time = time.time()
        for lead_time in range(31):
            # CRPS distribution should be dependent on variable
            if var in [4]:
                loss = crps_cost_function_trunc
            else:
                loss = crps_cost_function
            num = num + 1
            EMOS_glob = build_EMOS_network_keras(compile=True, lr = lr, loss = loss)
            EMOS_glob.fit(
                [
                    X_train_var_lead_all[var][lead_time].isel(mean_std=0).values.flatten(),
                    X_train_var_lead_all[var][lead_time].isel(mean_std=1).values.flatten(),
                ],
                y_train_var_lead_all[var][lead_time].values.flatten(),
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
            )
            EMOS_glob.save('/home/dchen/BA_CH_EN/models/EMOS_global_models/normed/EMOS_glob_' + var_names[var] + '_lead_time_' + str(lead_time) + '_normed.h5')
            
        # Printing out time
        end_time = time.time()
        time_difference = end_time - start_time
        hours = int(time_difference // 3600)
        minutes = int((time_difference % 3600) // 60)
        seconds = int(time_difference % 60)
        formatted_time = f" Round {num} finished in:{hours} hours, {minutes} minutes, {seconds} seconds"
        print(formatted_time)
            

# Check if the script is being run directly
if __name__ == "__main__":
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Extract the arguments from command line
        batch_size = int(sys.argv[1])
        epochs = int(sys.argv[2])
        lr = float(sys.argv[3])
        validation_split = float(sys.argv[3])

        # Call the main function with the provided arguments
        main(batch_size = batch_size, epochs = epochs, lr = lr, validation_split = validation_split)
    else:
        # Call the main function with default values
        main()