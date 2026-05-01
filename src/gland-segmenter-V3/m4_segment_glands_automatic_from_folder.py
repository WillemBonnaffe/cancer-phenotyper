##############
## INITIATE ##
##############

## Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import pickle
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
import argparse

## Import modules
from f_wsi_reader import slide_path_to_tiles_at_coordinates
from f_wsi_reader import get_tile_map
from f_wsi_reader import get_tile_map_fast
from f_wsi_reader import read_slide
from f_wsi_reader import read_region
from f_wsi_reader import get_dimensions
from f_wsi_reader import apply_all_filters
from f_wsi_reader import apply_all_transforms
from f_wsi_reader import reassemble_tiles
from f_wsi_reader import select_patches_on_grid
from f_wsi_reader import get_tissue_mask
from f_extract_objects import extract_object_pixels
from f_extract_objects import polish
from f_extract_objects import inlay_objects
from f_extract_objects import get_bounding_box_square

#
###

################################
## FUNCTIONS ARGUMENT PARSING ##
################################

def parse_args():
    parser = argparse.ArgumentParser(description="Extract glands from WSIs.")
    parser.add_argument('--WSI_FOLDER', type=str, required=True, help='Path to folder containing WSIs')
    parser.add_argument('--PT_OUTPUT_FOLDER', type=str, required=True, help='Output path for .pt files')
    return parser.parse_args()

#
###

###############
## FUNCTIONS ##
###############

## numpy_to_device
## Goal:
## Format numpy array for model.
## Inputs:
## X (list of numpy arrays): Input array for the model (B, H, W, C).
## Outputs:
## tensors (list of torch tensors): Output tensor for model (B, C, H, W).
def numpy_to_device(X, tile_size, device):
    tensors = []
    for x in X:        
        x = cv2.resize(x, (tile_size, tile_size))
        x = torch.tensor(x)        
        x = x.swapaxes(2, 1).swapaxes(1, 0) # (H, W, C) --> (C, H, W)
        x = x.float()
        x = x[None, :, :, :] # (C, H, W) --> (B=1, C, H, W)
        x = x.to(device)
        tensors += [x]        
    return tensors

#
###

################
## PARAMETERS ##
################

## Parameters for inputs
WSI_FOLDER = ''# '/Volumes/Elements/BDI/datasets/ICGCC/images/svs/'

## Parameters for model application
LEVEL = 0 # Downsampling level 
TILE_SIZE_MODEL = 1024 # Size of tile for model
DEVICE = 'mps' # Device to load model, tiles, and perform computations
NUM_CLASSES = 3 # Number of outputs given by model

## Parameters for storing glands
BUFFER = 128 # Added distance around bounding box (in pixels)
PT_OUTPUT_FOLDER = ''# '/Volumes/Elements/BDI/projects/GSG/outputs/o1_extracted_glands/icgcc'

#
###

#############
## IF MAIN ##
#############

if __name__ == "__main__":
    args = parse_args()
    WSI_FOLDER = args.WSI_FOLDER
    PT_OUTPUT_FOLDER = args.PT_OUTPUT_FOLDER

#
###

###########################
## GET PATHS FROM FOLDER ##
###########################

# Walk through WSI_FOLDER to find all .svs files
WSI_PATHS = []
for root, dirs, files in os.walk(WSI_FOLDER):
    for file in files:
        if file.lower().endswith('.svs'):
            full_path = os.path.join(root, file)
            WSI_PATHS.append(full_path)

# Optional: sort the list for consistency
WSI_PATHS.sort()

# Print result
for path in WSI_PATHS:
    print(path)

if os.path.exists(PT_OUTPUT_FOLDER) == False:
    os.mkdir(PT_OUTPUT_FOLDER)

#
###

###########################
## LOAD GLAND MODEL DUAL ##
###########################

## Model classes
from f_model import UNet
from f_model_transformer import UNetTransformer

## Load model
device = "mps"
print(f"Using {DEVICE} device")
pt_model_in = 'models/UNet_V1_1.pth'
model_1 = UNet(n_classes=NUM_CLASSES).to(device)
if os.path.exists(pt_model_in):
    model_1.load_state_dict(torch.load(pt_model_in))
    print("Loaded model")

## Load model
device = "mps"
print(f"Using {DEVICE} device")
pt_model_in = 'models/TransUNet_V1_1.pth'
model_2 = UNetTransformer(n_classes=NUM_CLASSES).to(device)
if os.path.exists(pt_model_in):
    model_2.load_state_dict(torch.load(pt_model_in))
    print("Loaded model")

## Combine models
def model(X):
    return model_1(X) + model_2(X)

def device_to_numpy_gland(X, tile_size):
    images = []
    for x in X:
        x = torch.sigmoid(x) * 255 # Logits to probabilities to [0, 255]
        # x = (torch.sigmoid(x) > 0.75) * 255 # Logits to probabilities to [0, 255]
        # x = x * 255 # Logits to probabilities to [0, 255]
        x = x.cpu().numpy()
        x = x.swapaxes(0, 1).swapaxes(1, 2)    
        x = x.astype('uint8')
        x = cv2.resize(x, (tile_size, tile_size))
        # x = np.argmax(x, axis=2).reshape(x.shape[0], x.shape[1], 1) # Only for multiple classes
        images += [x]        
    return images

#
###

#############################
## LOAD NUCLEI MODEL TRUAL ##
#############################

## Load model
pt_model_in = 'models_nuclei/r13_1.pth'
model_nuclei_1 = UNet(n_classes=1).to(device)
if os.path.exists(pt_model_in):
    model_nuclei_1.load_state_dict(torch.load(pt_model_in))
    print("Loaded model")

## Load model
pt_model_in = 'models_nuclei/r13_2.pth'
model_nuclei_2 = UNet(n_classes=1).to(device)
if os.path.exists(pt_model_in):
    model_nuclei_2.load_state_dict(torch.load(pt_model_in))
    print("Loaded model")

## Load model
pt_model_in = 'models_nuclei/r13_3.pth'
model_nuclei_3 = UNet(n_classes=1).to(device)
if os.path.exists(pt_model_in):
    model_nuclei_3.load_state_dict(torch.load(pt_model_in))
    print("Loaded model")

## Combine models
def model_nuclei(X):
    return torch.sigmoid(model_nuclei_1(X)) * torch.sigmoid(model_nuclei_2(X)) * torch.sigmoid(model_nuclei_3(X))
    # return model_nuclei_1(X) + model_nuclei_2(X) + model_nuclei_3(X)

def device_to_numpy_nuclei(X, tile_size):
    images = []
    for x in X:
        # x = torch.sigmoid(x) * 255 # Logits to probabilities to [0, 255]
        # x = (torch.sigmoid(x) > 0.75) * 255 # Logits to probabilities to [0, 255]
        x = x * 255 # Logits to probabilities to [0, 255]
        # x = (x > 0.999) * 255 # Logits to probabilities to [0, 255]
        x = x.cpu().numpy()
        x = x.swapaxes(0, 1).swapaxes(1, 2)    
        x = x.astype('uint8')
        x = cv2.resize(x, (tile_size, tile_size))
        # x = np.argmax(x, axis=2).reshape(x.shape[0], x.shape[1], 1) # Only for multiple classes
        images += [x]        
    return images

#
###

##############
## INITIATE ##
##############

## Debug mode
DEBUG = False

## For each WSI
for WSI_PATH in WSI_PATHS:
    print(WSI_PATH)
    
    ## Paths
    # WSI_PATH = WSI_PATHS[0] # DEBUG
    PT_INPUT = f"{PT_OUTPUT_FOLDER}/{os.path.basename(WSI_PATH)}_bboxes.npy"
    
    ## Load slide and bounding boxes
    slide = read_slide(WSI_PATH)
    bbox_list = np.load(PT_INPUT)

    ## For all bounding boxes
    for i, bbox in enumerate(bbox_list):
    
        ## Iterator
        print(f"{i+1}/{len(bbox_list)}", end='\r')
    
        ## Get bounding box of the object
        x_min, y_min, x_max, y_max = bbox # get_bounding_box_square(rescaled_gland_coords[10])
        
        ## Get location and size
        location_xy = (x_min - BUFFER, y_min - BUFFER)
        tile_size_xy = (x_max - x_min + 2 * BUFFER, y_max - y_min + 2 * BUFFER)
        # print(tile_size_xy)
        
        ## get object
        object_img = read_region(slide, location_xy, LEVEL, (np.max([TILE_SIZE_MODEL, tile_size_xy[0]]), np.max([TILE_SIZE_MODEL, tile_size_xy[1]])))
        object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB)
        # print(object_img.shape)
        
        ## Glands
        with torch.no_grad():
            X = numpy_to_device([object_img], TILE_SIZE_MODEL, DEVICE)
            object_mask_gland = device_to_numpy_gland(model(X[0]),np.max([TILE_SIZE_MODEL, tile_size_xy[0]]))[0]
        
        ## Nuclei
        with torch.no_grad():
            X = numpy_to_device([object_img], TILE_SIZE_MODEL, DEVICE)
            object_mask_nuclei = device_to_numpy_nuclei(model_nuclei(X[0]),np.max([TILE_SIZE_MODEL, tile_size_xy[0]]))[0]
        
        ## Crop
        object_img = object_img[0:tile_size_xy[0],0:tile_size_xy[1]]
        object_mask_gland = object_mask_gland[0:tile_size_xy[0],0:tile_size_xy[1]]
        object_mask_nuclei = object_mask_nuclei[0:tile_size_xy[0],0:tile_size_xy[1]]
        # print(object_img.shape)
        
        ## Combine masks
        # combined_mask = (0.5 * (object_mask_gland + object_mask_nuclei[:,:,None])).astype(np.uint8)
        combined_mask = np.copy(object_mask_gland)
        #
        ## Version 1
        # combined_mask = combined_mask[:,:,[2,1,0]]
        # combined_mask[:,:,0] = object_mask_nuclei
        #
        ## Version 2
        combined_mask = combined_mask[:,:,[2,1,0]]
        # combined_mask[:,:,0] = object_mask_nuclei
        combined_mask[:,:,2] = object_mask_nuclei
        
        ## Save the object image    
        object_filename = os.path.join(PT_OUTPUT_FOLDER, f'{os.path.basename(WSI_PATH)}__{x_min}_{x_max}_{y_min}_{y_max}_raw.png')
        cv2.imwrite(object_filename, object_img)
        
        ## Save the object image    
        object_filename = os.path.join(PT_OUTPUT_FOLDER, f'{os.path.basename(WSI_PATH)}__{x_min}_{x_max}_{y_min}_{y_max}_mask_glands.png')
        cv2.imwrite(object_filename, object_mask_gland)
        
        ## Save the object image    
        object_filename = os.path.join(PT_OUTPUT_FOLDER, f'{os.path.basename(WSI_PATH)}__{x_min}_{x_max}_{y_min}_{y_max}_mask_nuclei.png')
        cv2.imwrite(object_filename, object_mask_nuclei)
        
        ## Save the object image    
        object_filename = os.path.join(PT_OUTPUT_FOLDER, f'{os.path.basename(WSI_PATH)}__{x_min}_{x_max}_{y_min}_{y_max}_mask_both.png')
        cv2.imwrite(object_filename, combined_mask)
    
        if DEBUG:
            break
    
#
###
