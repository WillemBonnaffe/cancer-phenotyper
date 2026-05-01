#!/usr/bin/env python3

##############
## INITIATE ##
##############

## Imports
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import cv2
import matplotlib.pyplot as plt
import argparse

# Import custom modules for handling WSI dataset and encoder
from p_comp2d.f_dataset_wsi import WSIDatasetByInstance
from p_comp2d.f_dataset_wsi import display_wsi_tiles
from p_comp2d.f_pretrain import train as train_contrastive  # Updated to use the contrastive learning approach we've developed
from p_comp2d.f_train import evaluate_model
from p_comp2d.f_utils import format_path_file, format_label_file

#
###

#########################
## USER DEFINED INPUTS ##
#########################

## Argument parser
parser = argparse.ArgumentParser(description="Run encoder evaluation script.")
parser.add_argument('--pt_input_folder', type=str, required=True, help='Path to input folder')
parser.add_argument('--tile_size', type=int, default=256, help='Tile size')
parser.add_argument('--level', type=int, default=0, help='Magnification level')
parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--device', type=str, default='mps', help='Device (e.g., "cpu", "cuda", "mps")')
parser.add_argument('--encoder_type', type=str, choices=["resnet18", "resnet50", "vit"], default="vit", help='Type of encoder')

args = parser.parse_args()

## Architecture
from p_comp2d.f_encoder import ResNet18FeatureExtractor, ResNet50FeatureExtractor, ViTFeatureExtractor
if args.encoder_type == "resnet18":
    EncoderArchitecture = BasicResNet18Encoder
elif args.encoder_type == "resnet50":
    EncoderArchitecture = ResNet50FeatureExtractor
else:
    EncoderArchitecture = ViTFeatureExtractor

## Paths
pt_input_folder = args.pt_input_folder
pt_subfolder_out = pt_input_folder + 'encoder/'
pt_encoder_in = pt_subfolder_out + 'mod_enc_pretrained.pt'
pt_path_file = pt_input_folder + 'tab_paths.txt'
pt_label_file = pt_input_folder + 'tab_labels.txt'
pt_tile_coordinates_list = pt_input_folder + 'preprocessed/obj_coordinates.pkl'

tile_size = args.tile_size
level = args.level
embedding_dim = args.embedding_dim
batch_size = args.batch_size
device = args.device

#
###

#################
## LOAD INPUTS ##
#################

## Create output folder if it doesn't exist
if not os.path.exists(pt_subfolder_out):
    os.mkdir(pt_subfolder_out)

## Load WSI paths and labels
wsi_path_list = format_path_file(pt_path_file)
labels = format_label_file(pt_label_file)

## Load coordinates
tile_coordinate_list = pickle.load(open(pt_tile_coordinates_list, "rb"))

## Number of labels
num_labels = len(labels[0])

## Checks
print(num_labels)
print(len(labels))
print(len(wsi_path_list))

#
###

##################
## LOAD ENCODER ##
##################

## Create an instance of the encoder with the desired embedding dimension
encoder = EncoderArchitecture()
# encoder = EncoderArchitecture(embedding_dim)

## Load pre-trained model if it exists
if os.path.exists(pt_encoder_in):
    encoder.load_state_dict(torch.load(pt_encoder_in))
    print("Loaded pre-trained model.")
encoder = encoder.to(device)

#
###

######################
## EVALUATE ENCODER ##
######################

t0 = time.time()
embeddings = []
ground_truth = []
for i in range(len(wsi_path_list)):

    print(f"sample: {i+1}/{len(wsi_path_list)}")
    print(f"path: " + wsi_path_list[i])

    ## Prepare data for evaluation
    t0_ = time.time()
    all_dataset = WSIDatasetByInstance([wsi_path_list[i]], [tile_coordinate_list[i]], [labels[i]],
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
pt_embeddings_out = pt_input_folder + "encoder/" + "obj_embeddings.pkl"
with open(pt_embeddings_out, "wb") as file:
    pickle.dump(embeddings, file)

#
###
