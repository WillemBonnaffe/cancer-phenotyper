#####
## ##
#####

##############
## INITIATE ##
##############

## Imports
import os
import numpy as np

#
###

#####################
## UTILS FUNCTIONS ##
#####################

def collapse_columns_2Darray_to_string(arr, sep=""):
    """
    """
    arr_str = []
    for arr_ in arr:
        arr_ = [arr__ for arr__ in arr_.astype(str)]
        arr_ = sep.join(arr_)
        arr_str += [arr_]
    return np.array(arr_str)

def subset_list(list_, indice_array):
    """
    """
    return [list_[i] for i in indice_array]

def format_path_file(wsi_paths_file):
    """
    """
    ## Read the WSI paths from the file
    with open(wsi_paths_file, 'r') as f:
        wsi_path_list = [line.strip() for line in f]

    return wsi_path_list

def format_label_file(label_file):
    """
    """
    ## Read labels
    labels = np.genfromtxt(label_file)

    ## Format labels
    if len(labels.shape) == 1: 
        labels = labels.reshape(-1,1)

    return labels

def normalise(x):
    return (x - x.min())/(x.max() - x.min())

def subset_and_concatenate(arr_list, max_num_instances):
    """
    """
    subset_arr_list = []
    index_list = []
    k = 0
    for arr in arr_list:
        num_instances = arr.shape[0]
        subset_indices = np.random.choice(num_instances, size=min(num_instances, max_num_instances), replace=False)
        subset_arr = arr[subset_indices]
        subset_arr_list.append(subset_arr)
        index = np.repeat(k, min(num_instances, max_num_instances))
        index_list.append(index)
        k = k + 1
    concatenated_arr = np.concatenate(subset_arr_list, axis=0)
    concatenated_index = np.concatenate(index_list, axis=0)
    return concatenated_arr, concatenated_index

#
###
