import os
import numpy as np
import fnmatch

def crps_load_lead_lat_lon(var_name):
    '''
    Args:
        path (string): path to the directory that contains the numpy arrays
        var_name (string): the variable name used in the numpy files
    Returns:
        list of numpy arrays
    '''
    path = "/Data/Delong_BA_Data/scores/crps_benchmark_scores/"
    # Create the file pattern based on the variable name
    file_pattern = f'{var_name}_lead_*_scores.npy'

    # List all files in the directory
    files = os.listdir(path)

    # Filter the list to only include .npy files that match the file pattern
    npy_files = [file for file in files if fnmatch.fnmatch(file, file_pattern)]
    
    # Sort the file list based on the lead time
    npy_files.sort(key=lambda file: int(file.split('_')[2]))

    # Load each .npy file and store it in a list
    arrays = [np.load(os.path.join(path, file)) for file in npy_files]

    return arrays


def crps_load_lead_lat_lon_t2m():
    '''
    Function to load t2m crps scores
    '''
    return crps_load_lead_lat_lon('t2m')


def crps_load_lead_lat_lon_ws10():
    '''
    Function to load ws10 crps scores
    '''
    return crps_load_lead_lat_lon('ws10')                                  