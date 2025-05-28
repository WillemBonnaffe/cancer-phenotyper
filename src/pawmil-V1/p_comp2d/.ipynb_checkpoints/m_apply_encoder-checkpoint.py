##### 
## ##
#####

## WIPs:
## WIP-23.04.2024.1: error if only one slide
## WIP-23.04.2024.2: error if level not 0

##############
## INITIATE ##
##############

## Imports
import os
import numpy as np
import pickle
import cv2
import time
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

## Import modules
from .f_dataset_wsi import WSIDatasetByInstance
from .f_encoder import ResNet50FeatureExtractor as EncoderArchitecture
# from .f_encoder import BasicResNet18Encoder as EncoderArchitecture
from .f_train import evaluate_model
from .f_utils import format_path_file
from .f_utils import format_label_file

#
###

###############
## FUNCTIONS ##
###############

def main(pt_input_folder, paths_file, labels_file, tile_size, level, device):
    """
    """
    #############################
    ## USER DEFINED PARAMETERS ##

    ## Paths
    pt_folder = pt_input_folder 
    pt_subfolder_in = pt_folder + f'encoder/'
    pt_subfolder_out = pt_folder + f'encoder/'
    pt_encoder_in = pt_subfolder_in + 'mod_enc.pt'
    pt_embeddings_out = pt_subfolder_out + 'obj_embeddings.pkl'
    pt_tile_coordinates_list = pt_folder + 'preprocessed/obj_coordinates.pkl'
    pt_path_file = pt_folder + paths_file 
    pt_label_file = pt_folder + labels_file 

    ## Wsi parameters
    tile_size = tile_size
    level = level

    ## Evaluation parameters
    batch_size = 16
    device = device

    ##############
    ## INITIATE ##

    ## Create output subfolder
    if os.path.exists(pt_subfolder_out) == False:
        os.mkdir(pt_subfolder_out)

    ## Get wsi path list and labels and coordinates
    wsi_path_list = format_path_file(pt_path_file)
    labels = format_label_file(pt_label_file)
    tile_coordinate_list = pickle.load(open(pt_tile_coordinates_list, "rb"))
    print("loaded coordinates")

    ## 
    num_labels = len(labels[0])
  
    ######################
    ## EVALUATE ENCODER ##

    print("evaluating encoder...")

    ## Instantiate the model
    encoder = EncoderArchitecture()
    if os.path.exists(pt_encoder_in):
        encoder.load_state_dict(torch.load(pt_encoder_in))
    else:
        torch.save(encoder.state_dict(), pt_encoder_in)

    t0 = time.time()
    embeddings = []
    ground_truth = []
    for i in range(len(wsi_path_list)):

        print(f"sample: {i+1}/{len(wsi_path_list)}")
        print(f"path: " + wsi_path_list[i])

        ## Prepare data for evaluation
        t0_ = time.time()
        all_dataset = WSIDatasetByInstance([wsi_path_list[i]], [tile_coordinate_list[i]], [labels[i]], # WIP-23.04.2024.1: error if only one slide -> to fix
                tile_size=tile_size, level=level, preload_tiles=True, num_labels=num_labels, augment=True)
        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
        tf_ = time.time()
        print(f"loaded tiles in {tf_-t0_:.2f}s")

        ## Evaluate
        t0_ = time.time()
        outputs_, embeddings_, _, ground_truth_ = evaluate_model(encoder, all_loader, device)
        tf_ = time.time()
        print(f"evaluated encoder in {tf_-t0_:.2f}s")

        ## Store embeddings
        embeddings.append(embeddings_)
        ground_truth.append(ground_truth_)

    tf = time.time()
    print(f"generated embeddings in {tf-t0:.2f}s")

    ## Format embeddings and save
    with open(pt_embeddings_out, "wb") as file:
        pickle.dump(embeddings, file)

    ## Save coordinates of embeddings
    with open(pt_subfolder_out + "obj_coordinates.pkl", "wb") as file:
        pickle.dump(tile_coordinate_list, file)

#
###
