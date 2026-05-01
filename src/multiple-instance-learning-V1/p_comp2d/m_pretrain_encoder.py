import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import custom modules for handling WSI dataset and encoder
from .f_dataset_wsi import WSIDatasetByInstance
from .f_encoder import BasicResNet18Encoder as EncoderArchitecture
from .f_train_contrastive import train_contrastive  # Updated to use the contrastive learning approach we've developed
from .f_utils import format_path_file, format_label_file

def main(pt_input_folder, tile_size, level, num_features, num_epochs, batch_size, device):
    """
    Pretrain an encoder using contrastive learning on a WSI dataset.
    
    Args:
        pt_input_folder (str): Path to the folder containing WSI tiles and related files.
        tile_size (int): Size of the tiles in pixels.
        level (int): Downsampling level for WSI.
        num_features (int): Number of features per instance (embedding dimension).
        num_epochs (int): Number of epochs for pretraining.
        batch_size (int): Batch size for data loading.
        device (str): Device on which to perform the training (e.g., 'cpu' or 'cuda').
    """

    ## Start time
    t0 = time.time()

    #########################
    ## USER DEFINED INPUTS ##

    ## Paths
    pt_input_folder = pt_input_folder 
    pt_subfolder_out = pt_input_folder + "encoder/"
    pt_encoder_in = pt_subfolder_out + "mod_enc.pt"
    pt_path_file = pt_input_folder + "wsi_paths.txt"  # Assuming a text file listing WSI paths
    pt_label_file = pt_input_folder + "wsi_labels.txt"  # Assuming a text file listing WSI labels

    ## Parameters
    tile_size = tile_size
    level = level
    embedding_dim = num_features
    num_epochs = num_epochs
    batch_size = batch_size
    device = device

    ## Fixed parameters for contrastive learning
    projection_dim = 64
    trainval_split_prop = 0.75

    ##############
    ## INITIATE ##

    ## Create output folder if it doesn't exist
    if not os.path.exists(pt_subfolder_out):
        os.mkdir(pt_subfolder_out)

    ## Load WSI paths and labels
    wsi_path_list = format_path_file(pt_path_file)
    labels = format_label_file(pt_label_file)

    ## Adjust labels for unsupervised training (e.g., using sample indices)
    labels = np.arange(len(wsi_path_list))
    trainval_split = int(trainval_split_prop * len(wsi_path_list))

    ###############
    ## LOAD DATA ##

    ## Load WSI dataset
    train_dataset = WSIDatasetByInstance(wsi_path_list[:trainval_split], labels[:trainval_split], level=level, 
                                         tile_size=tile_size, preload_tiles=True, augment=True)
    test_dataset = WSIDatasetByInstance(wsi_path_list[trainval_split:], labels[trainval_split:], level=level, 
                                        tile_size=tile_size, preload_tiles=True, augment=False)

    ## Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ##################
    ## LOAD ENCODER ##

    ## Create an instance of the encoder with the desired embedding dimension
    encoder = EncoderArchitecture(embedding_dim)

    ## Load pre-trained model if it exists
    if os.path.exists(pt_encoder_in):
        encoder.load_state_dict(torch.load(pt_encoder_in))
        print("Loaded pre-trained model.")
    encoder.to(device)

    ################
    ## TRAIN LOOP ##
    
    ## Pretrain encoder using the contrastive learning approach
    encoder, train_loss, test_loss = train_contrastive(encoder, train_loader, test_loader, embedding_dim, projection_dim, num_epochs, device)
    
    ## Save the fine-tuned model
    torch.save(encoder.state_dict(), pt_encoder_in)
    
    ## End time 
    tf = time.time()
    print(f"Time running: {tf-t0:.2f}s")

#
###

