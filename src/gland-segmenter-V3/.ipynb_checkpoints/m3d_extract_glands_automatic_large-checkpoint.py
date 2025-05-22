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

#####################################
## FUNCTIONS FOR MODEL APPLICATION ##
#####################################

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

## device_to_numpy
## Goal: 
## Format tensor on device back to numpy array.
## Inputs:
## X (list of torch tensors): List of torch tensors predictions from the model (B, C, H, W).
## Outputs:
## images (list of numpy arrays): List of images produced by the model after re-formatting (B, H, W, C).
def device_to_numpy(X, tile_size):
    images = []
    for x in X:
        x = torch.sigmoid(x) * 255 # Logits to probabilities to [0, 255]
        x = x.cpu().numpy()
        x = x.swapaxes(0, 1).swapaxes(1, 2)    
        x = x.astype('uint8')
        x = cv2.resize(x, (tile_size, tile_size))
        x = np.argmax(x, axis=2).reshape(x.shape[0], x.shape[1], 1) # Only for multiple classes
        images += [x]        
    return images

#
###

"""
###########
## PATHS ##
###########

## Paths TCGA FFPE
WSI_PATHS = [    
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/gdc_download_20241203_174927.283959/9f2c8275-5ff0-4167-9238-66effdd28e24/TCGA-HC-7213-01Z-00-DX1.5b03a3f2-ae77-4906-81b6-e9a5c0c5ed16.svs',
    '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/TCGA-HC-A6AP-01Z-00-DX1.1E2C19B4-6757-488D-AFB4-71AE5FD6EC11.svs',
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/gdc_download_20241203_175115.987620/eb00cbed-63c4-4d47-9b6a-9dde1306b8cd/TCGA-KK-A7B2-01Z-00-DX1.3E779031-6FE4-4BD0-838C-D9ED49E1B9A7.svs',
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/gdc_download_20241203_175456.004065/f4323e88-33d7-4e7d-9763-b15be4e73b28/TCGA-XJ-A83F-01Z-00-DX1.11A9B6FC-16AB-44F2-B93E-7806693D90F7.svs',
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/TCGA-HC-7209-01A-01-TS1.028552bb-3a9b-44dd-80c4-49d9680b2807.svs',
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/TCGA-HC-A6AQ-01A-01-TS1.CDFDB554-F9D7-4CB1-A4FF-48C18418AB65.svs',
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/TCGA-HC-A6AS-01A-01-TS1.A86A75AD-C6F2-4497-A955-393A302BE5DD.svs',
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/TCGA-J4-A67N-01A-01-TS1.697C38C8-0FE4-4D58-BD35-2B3510BB63D2.svs',
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/TCGA-SU-A7E7-01A-02-TSB.4501DDB2-75F8-46CB-A677-80981853C9FC.svs',
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/TCGA-YL-A8SK-01B-02-TSB.584CE622-2AE8-421A-855F-DB7CFBAE45AF.svs', # Bad quality tissue
    # '/Users/user/Documents/projects/GlandSeg/data/TCGA_FFPEs/TCGA-HC-7818-01Z-00-DX1.06895343-4b63-40b2-afbb-b31b262a4165.svs',
]

#
###
"""

################
## PARAMETERS ##
################

## Parameters for inputs
WSI_FOLDER = '' # '/Volumes/Elements/BDI/datasets/ICGCC/images/svs/'

## Parameters for tile extraction
TILE_SIZE = 64 # Size of tiles for tissue extraction
FILTER_LOW = 10 # Low filter to remove black tiles
FILTER_HIGH = 240 # High filter to remove white tiles
FILTER_STD = 10 # Std filter to remove grey tiles

## Parameters for model application
LEVEL = 0 # Downsampling level 
TILE_SIZE_MODEL = 1024 # Size of tile for model
DEVICE = 'mps' # Device to load model, tiles, and perform computations
# PT_MODEL_IN = 'models/TransUNet_V2_0.pth' # Path to model weights
NUM_CLASSES = 3 # Number of outputs given by model
BATCH_SIZE = 32 # Maximum number of patches loaded at one time
DOWNSAMPLE_TILES = 4

## Parameters for gland extraction
DOWNSAMPLE_EXTRACTION = 4 # Downsample factor of mask for extraction of objects
NUM_POLISH = 1 # Number of times to erode mask (reduced clumped objects but may lose small objects)

## Parameters for storing glands
BUFFER = 128 # Added distance around bounding box (in pixels)
THRESHOLD_MIN = 1000 # Minimum surface area of objects (in pixels)
THRESHOLD_MAX = 40000000 # Maximum surface area of objects (in pixels)
PROP_BLACK = 0.0 # Maximum proportion of border pixels in image (proportion)
PT_OUTPUT_FOLDER = '' # '/Volumes/Elements/BDI/projects/GSG/outputs/o1_extracted_glands/icgcc'
# PT_OUTPUT_FOLDER = '/Users/user/Documents/projects/GlandSeg/outputs/V2/o1_extracted_glands/TCGA-HC-A6AP-01Z-00-DX1.1E2C19B4-6757-488D-AFB4-71AE5FD6EC11.svs' # Path where the gland images should be saved

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

#
###

#####################
## LOAD MODEL DUAL ##
#####################

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

#
###

##################################
## EXTRACT GLANDS AUTOMATICALLY ##
##################################

## Debug model
DEBUG = False

## For each WSI
for WSI_PATH in WSI_PATHS:
    print(WSI_PATH)
    

    ##################
    ## GET TILE MAP ##
    ##################
    
    ## Open slide
    slide = read_slide(WSI_PATH)
    slide_size = get_dimensions(slide)
    
    ## Get tile_map
    thumbnail, tile_map, canvas_size = get_tile_map_fast(slide, TILE_SIZE) # Canvas is a space containing the WSI.
    
    ## Rescale canvas size
    canvas_size_ds = canvas_size // DOWNSAMPLE_TILES
    
    #
    ###
    
    ## Checks
    if DEBUG:
        plt.figure(figsize=(15,15))
        plt.imshow(thumbnail)
        plt.show()
        plt.close()
    
    ##########################
    ## GET TILE COORDINATES ##
    ##########################
    
    ## Get tissue mask
    tile_mask = get_tissue_mask(tile_map, FILTER_LOW, FILTER_STD, FILTER_HIGH, threshold=10000)
    
    ## Get coordinates of tiles with tissue
    tile_coordinates = (np.argwhere(tile_mask[:,:,0] > 0) * TILE_SIZE).astype(int)
    tile_coordinates = [[tile_coordinate[0], tile_coordinate[1]] for tile_coordinate in tile_coordinates] # (x: H, y: W)
    
    #
    ###
    
    ## Checks
    if DEBUG:
        plt.figure(figsize=(15,15))
        plt.imshow(tile_mask)
        plt.show()
        plt.close()
        
    ###############################
    ## EXTRACT PATCHES FOR MODEL ##
    ###############################
    
    ## Remove patches too close to the edges (WIP TO REMOVE)
    tile_coordinates_thinned = []
    for tile_coordinates_ in tile_coordinates:
        coords_0_lo = tile_coordinates_[0] + int(TILE_SIZE/2) - int(TILE_SIZE_MODEL/2)
        coords_0_hi = tile_coordinates_[0] + int(TILE_SIZE/2) + int(TILE_SIZE_MODEL/2)
        coords_1_lo = tile_coordinates_[1] + int(TILE_SIZE/2) - int(TILE_SIZE_MODEL/2)
        coords_1_hi = tile_coordinates_[1] + int(TILE_SIZE/2) + int(TILE_SIZE_MODEL/2)
        if (coords_0_lo > 0) & (coords_0_hi < canvas_size[0]) & (coords_1_lo > 0) & (coords_1_hi < canvas_size[1]):
            tile_coordinates_thinned += [[coords_0_lo, coords_1_lo]]
    
    ## Get patches closest to grid points (WIP TO MAKE FASTER)
    tile_coordinates_thinned = select_patches_on_grid(tile_coordinates_thinned, grid_spacing=512)
    print(f'Number of patches: {len(tile_coordinates_thinned)}')
    
    ## Rescale coordinates according to downsampling factor
    tile_coordinates_thinned_downsampled = [[coords[0] // DOWNSAMPLE_TILES, coords[1] // DOWNSAMPLE_TILES] for coords in tile_coordinates_thinned]
    
    #
    ###
    
    ## Checks
    if DEBUG:
        for i in range(0, len(tile_coordinates_thinned), BATCH_SIZE):
            
            ## Load patches batch
            tile_coordinates_thinned_ = tile_coordinates_thinned[i:i + BATCH_SIZE]
            patches_ = slide_path_to_tiles_at_coordinates(WSI_PATH, tile_coordinates_thinned_, TILE_SIZE_MODEL, LEVEL)
            plt.imshow(patches_[0])
            plt.show()
            plt.imshow(patches_[1])
            plt.show()
            break
    
    ################# 
    ## APPLY MODEL ##
    #################
    
    ## Apply to all tiles
    patches = []
    preds = []
    for i in range(0, len(tile_coordinates_thinned), BATCH_SIZE):
        
        ## Load patches batch
        tile_coordinates_thinned_ = tile_coordinates_thinned[i:i + BATCH_SIZE]
        patches_ = slide_path_to_tiles_at_coordinates(WSI_PATH, tile_coordinates_thinned_, TILE_SIZE_MODEL, LEVEL)
    
        ## Apply model
        X = numpy_to_device(patches_, TILE_SIZE_MODEL, DEVICE)
        with torch.no_grad():
            for x in X:                
                preds_ = model(x)        
                preds_ = device_to_numpy(preds_, TILE_SIZE_MODEL // DOWNSAMPLE_TILES)[0] # WiP: Only for single patch            
                preds += [preds_] # WiP: Only for single patch
    
        ## Reassemble downsampled WSI
        for patches__ in patches_:
            patches__ = cv2.resize(patches__, (TILE_SIZE_MODEL // DOWNSAMPLE_TILES, TILE_SIZE_MODEL // DOWNSAMPLE_TILES))
            patches += [patches__]
    
    #
    ###
    
    ####################
    ## REASSEMBLE WSI ##
    ####################
    
    ## Reassemble image
    img = reassemble_tiles(patches, tile_coordinates_thinned_downsampled, canvas_size_ds[0], canvas_size_ds[1])
    
    ## Compute slide-level tissue mask
    tissue_mask = cv2.resize(tile_mask, (canvas_size_ds[1], canvas_size_ds[0]), cv2.INTER_LINEAR)[:,:,None] # (H, W)
    
    ## Mask WSI
    img = img * tissue_mask
    
    #
    ###
    
    ## Checks
    if DEBUG:
        plt.figure(figsize=(15,15))
        plt.imshow(img)
        plt.show()
        plt.close()
    
    ############################
    ## REASSEMBLE PREDICTIONS ##
    ############################
    
    ## Assemble predictions
    mask = reassemble_tiles(preds, tile_coordinates_thinned_downsampled, canvas_size_ds[0], canvas_size_ds[1])
    mask = mask[:,:,0].reshape(mask.shape[0], mask.shape[1], 1) * tissue_mask
    
    ## Keep full size mask
    mask_full_size = mask
    
    #
    ###
    
    ## Checks
    if DEBUG:
        plt.figure(figsize=(15,15))
        plt.imshow(mask)
        plt.show()
        plt.close()
    
    ########################################
    ## PREPARE MASK FOR OBJECT EXTRACTION ##
    ########################################
    
    ## Binarise mask
    mask = ((mask > 0.5)*1).astype('uint8')
    mask = cv2.resize(mask, (mask.shape[1] // DOWNSAMPLE_EXTRACTION, mask.shape[0] // DOWNSAMPLE_EXTRACTION))
    mask = mask[:,:,None]
    
    ## Polish mask
    for i in range(NUM_POLISH):
        mask = polish(mask)
        mask = mask[:,:,None]
    
    ## Downscale tissue mask
    tissue_mask_downsampled = cv2.resize(tissue_mask, (tissue_mask.shape[1] // DOWNSAMPLE_EXTRACTION, tissue_mask.shape[0] // DOWNSAMPLE_EXTRACTION))
    tissue_mask_downsampled = tissue_mask_downsampled[:,:,None]
    
    #
    ###
    
    ## Checks
    if DEBUG:
        plt.figure(figsize=(15,15))
        plt.imshow(mask + tissue_mask_downsampled)
        plt.show()
        plt.close()
    
    #####################
    ## EXTRACT OBJECTS ##
    #####################
    
    ## Get objects
    gland_coords = extract_object_pixels(mask)
    print(f'Number of glands: {len(gland_coords)}')
    
    #
    ###
    
    ## Checks
    if DEBUG:
        fig, axes = plt.subplots(1, 1, figsize=(15,15))
        axes.imshow(inlay_objects(mask, gland_coords))
        plt.show()
        plt.close()
    
    ####################
    ## FILTER OBJECTS ##
    ####################
    
    ## Criteria for filtering
    buffer = BUFFER // DOWNSAMPLE_EXTRACTION # Added distance around bounding box (in pixels)
    threshold_min = THRESHOLD_MIN // (DOWNSAMPLE_EXTRACTION ** 2) // (DOWNSAMPLE_TILES ** 2) # Minimum surface area of objects (in pixels)
    threshold_max = THRESHOLD_MAX // (DOWNSAMPLE_EXTRACTION ** 2) // (DOWNSAMPLE_TILES ** 2) # Maximum surface area of objects (in pixels)
    
    ## Extract and save each object
    gland_coords_thinned = []
    for i, coords in enumerate(gland_coords):
    
        ## Discard small objects
        if (len(coords) > threshold_min) & (len(coords) < threshold_max):
            
            ## Get bounding box of the object
            x_min, y_min, x_max, y_max = get_bounding_box_square(gland_coords[i])
    
            ## Get object
            object_img = mask[(x_min - buffer):(x_max + buffer), (y_min - buffer):(y_max + buffer)]
            object_tissue = tissue_mask_downsampled[(x_min - buffer):(x_max + buffer), (y_min - buffer):(y_max + buffer)]
            
            ## Check size
            if (object_img.shape[0] > 0) & (object_img.shape[1] > 0):
                
                if object_tissue.sum()/(object_tissue.shape[0] * object_tissue.shape[1]) >= (1 - PROP_BLACK):                    
            
                    ## Save the object coordinates
                    gland_coords_thinned += [gland_coords[i]]
    
    ## Check final number of glands
    print(f'Final number of glands: {len(gland_coords_thinned)}')
    
    ## Rescale coordinates to original size
    rescaled_gland_coords = [coords * DOWNSAMPLE_EXTRACTION * DOWNSAMPLE_TILES for coords in gland_coords_thinned]
    
    #
    ###
    
    ## Checks
    if DEBUG:
        fig, axes = plt.subplots(1, 1, figsize=(15,15))
        axes.imshow(inlay_objects(mask, gland_coords_thinned))
        plt.show()
        plt.close()
    
    ##################
    ## SAVE OBJECTS ##
    ##################
    
    ## Paths
    os.makedirs(PT_OUTPUT_FOLDER, exist_ok=True)
    
    ## Extract and save each object
    bbox_list = []
    identifiers_list = []
    for i, coords in enumerate(rescaled_gland_coords):
            
        ## Get bounding box of the object
        x_min, y_min, x_max, y_max = get_bounding_box_square(coords)
    
        ## Get object
        location_xy = (x_min - BUFFER, y_min - BUFFER)
        tile_size_xy = (x_max - x_min + 2 * BUFFER, y_max - y_min + 2 * BUFFER)
        object_img = read_region(slide, location_xy, LEVEL, tile_size_xy)
        object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB)
    
        ## Save the object image    
        object_filename = os.path.join(PT_OUTPUT_FOLDER, f'{os.path.basename(WSI_PATH)}__{x_min}_{x_max}_{y_min}_{y_max}.png')
        cv2.imwrite(object_filename, object_img)
    
        ## Save
        bbox_list += [[x_min, y_min, x_max, y_max]]
        identifiers_list += [object_filename]
    
    ## Save bbox
    bbox_filename = os.path.join(PT_OUTPUT_FOLDER, f'{os.path.basename(WSI_PATH)}_bboxes')
    np.save(bbox_filename, bbox_list)

    ## Save identifiers
    identifiers_filename = os.path.join(PT_OUTPUT_FOLDER, f'{os.path.basename(WSI_PATH)}_identifiers')
    np.save(identifiers_filename, identifiers_list)

    ## Save gland polygons
    pol_filename = os.path.join(PT_OUTPUT_FOLDER, f'{os.path.basename(WSI_PATH)}_polygons.pkl')
    with open(pol_filename, 'wb') as f:
        pickle.dump(gland_coords_thinned, f)
        
    #
    ###
    
#
###