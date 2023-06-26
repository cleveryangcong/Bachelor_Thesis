import os
import numpy as np
import fnmatch

def DRN_load_score_10(var_name):
    '''
    Args:
        path (string): path to the directory that contains the numpy arrays
        var_name (string): the variable name used in the numpy files
    Returns:
        list of numpy arrays
    '''
    path = "/Data/Delong_BA_Data/scores/DRN_scores_10/"
    # Create the file pattern based on the variable name
    file_pattern = f'DRN_{var_name}_lead_time_*_scores.npy'

    # List all files in the directory
    files = os.listdir(path)

    # Filter the list to only include .npy files that match the file pattern
    npy_files = [file for file in files if fnmatch.fnmatch(file, file_pattern)]
    
    # Sort the file list based on the lead time
    npy_files.sort(key=lambda file: int(file.split('_')[4]))
    
    # Load each .npy file and store it in a list
    arrays = [np.load(os.path.join(path, file)) for file in npy_files]

    return arrays