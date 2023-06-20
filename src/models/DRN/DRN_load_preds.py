import os
import numpy as np
import fnmatch

def DRN_load_preds(var_name):
    '''
    Args:
        var_name (string): the variable name used in the numpy files
    Returns:
        list of numpy arrays
    '''
    path = "/Data/Delong_BA_Data/preds/DRN_preds/"
    # Create the file pattern based on the variable name
    file_pattern = f'DRN_{var_name}_lead_*_preds.npy'

    # List all files in the directory
    files = os.listdir(path)

    # Filter the list to only include .npy files that match the file pattern
    npy_files = [file for file in files if fnmatch.fnmatch(file, file_pattern)]
    
    # Sort the file list based on the lead time
    npy_files.sort(key=lambda file: int(file.split('_')[4]))
    
    # Load each .npy file and store it in a list
    arrays = [np.load(os.path.join(path, file)).reshape((357, 120, 130, 2)) for file in npy_files]

    return arrays