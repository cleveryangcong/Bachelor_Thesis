import os
import numpy as np
import fnmatch

def u_net_load_score(var_name):
    '''
    Args:
        path (string): path to the directory that contains the numpy arrays
        var_name (string): the variable name used in the numpy files
    Returns:
        list of numpy arrays
        
    '''
    path = "/Data/Delong_BA_Data/scores/U_net/"
    
    dict_names = {'u10': 0, 'v10': 1, 't2m': 2, 't850':3, 'z500':4, 'ws10':5}
    var_num = dict_names[var_name]
    # Create the file pattern based on the variable name
    file_pattern = f'U_net_var_{var_num}_lead_*_scores.npy'

    # List all files in the directory
    files = os.listdir(path)

    # Filter the list to only include .npy files that match the file pattern
    npy_files = [file for file in files if fnmatch.fnmatch(file, file_pattern)]
    
    # Sort the file list based on the lead time
    npy_files.sort(key=lambda file: int(file.split('_')[5]))
    
    # Load each .npy file and store it in a list
    arrays = [np.load(os.path.join(path, file)) for file in npy_files]

    return arrays

def u_net_load_score_mean(var_name):
    '''
    Args:
        path (string): path to the directory that contains the numpy arrays
        var_name (string): the variable name used in the numpy files
    Returns:
        list of numpy arrays
        
    '''
    path = "/Data/Delong_BA_Data/scores/U_net_5_mean/"
    
    dict_names = {'u10': 0, 'v10': 1, 't2m': 2, 't850':3, 'z500':4, 'ws10':5}
    var_num = dict_names[var_name]
    # Create the file pattern based on the variable name
    file_pattern = f'U_net_mean_var_{var_num}_lead_*_scores.npy'

    # List all files in the directory
    files = os.listdir(path)

    # Filter the list to only include .npy files that match the file pattern
    npy_files = [file for file in files if fnmatch.fnmatch(file, file_pattern)]
    
    # Sort the file list based on the lead time
    npy_files.sort(key=lambda file: int(file.split('_')[6]))
    
    # Load each .npy file and store it in a list
    arrays = [np.load(os.path.join(path, file)) for file in npy_files]

    return arrays