import os
import numpy as np
import fnmatch

def EMOS_global_load_score(var_name):
    '''
    Args:
        path (string): path to the directory that contains the numpy arrays
        var_name (string): the variable name used in the numpy files
    Returns:
        list of numpy arrays
    '''
    path = "/Data/Delong_BA_Data/scores/EMOS_global_scores/"
    # Create the file pattern based on the variable name
    file_pattern = f'EMOS_global_{var_name}_lead_*_scores.npy'

    # List all files in the directory
    files = os.listdir(path)

    # Filter the list to only include .npy files that match the file pattern
    npy_files = [file for file in files if fnmatch.fnmatch(file, file_pattern)]

    # Load each .npy file and store it in a list
    arrays = [np.load(os.path.join(path, file)) for file in npy_files]

    return arrays

def EMOS_global_load_score_t2m():
    '''
    Function to load t2m EMOS global scores
    '''
    return EMOS_global_load_score('t2m')


def EMOS_global_load_score_ws10():
    '''
    Function to load ws10 EMOS global scores
    '''
    return EMOS_global_load_score('ws10')                                  