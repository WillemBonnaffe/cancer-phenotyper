#####
## ##
#####

## WIPs:
## WIP-23.04.2024: Update to be compatible with isyntax.

##############
## INITIATE ##
##############

## Imports
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

## Import modules
from .f_wsi_reader import slide_path_to_tiles_at_coordinates

#
###

################################
## IMAGE PROCESSING FUNCTIONS ##
################################

def preprocess_image(img):
    """
    """
    img = img/255
    img = img.transpose((2,0,1))
    return img

def preprocess_image_resnet18(img):
    """
    """
    ## Parameters
    mean_resnet = np.array([0.485, 0.456, 0.406])
    sd_resnet = np.array([0.229, 0.224, 0.225])

    ## Clahe
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):
        img[:,:,i] = clahe.apply(img[:,:,i])

    ## Transform image
    img = img/255
    img = (img - mean_resnet)/sd_resnet
    img = cv2.resize(img, (224, 224))
    img = img.transpose((2,0,1)) 

    return img

def augment_image(img):
    """
    Apply random rotation and/or flip to the image.
    """
    # Random flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # Vertical flip

    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        img_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return img

def display_wsi_tiles(dataset, sample_index, num_tiles=16):
    """
    Display a 4x4 panel of tiles extracted from a WSI using the dataset.

    Parameters:
    - dataset: Instance of WSIDatasetByInstance.
    - sample_index: Index of the WSI sample from which to extract and display tiles.
    - num_tiles: Number of tiles to display (default 16).
    """    
    ## 
    selected_indices = np.where(dataset.wsi_path_list[sample_index] == np.array(dataset.loaded_wsi_paths))[0][0:num_tiles]
    
    # Load the selected tiles using the dataset
    tiles = []
    for idx in selected_indices:
        tile, _ = dataset[idx]  # We can ignore the label here
        tiles.append(tile)
    
    # Convert tiles from (C, H, W) to (H, W, C) for visualization
    tiles = [tile.transpose(1, 2, 0) for tile in tiles]

    # Parameters for reverse standardization
    mean_resnet = np.array([0.485, 0.456, 0.406])
    sd_resnet = np.array([0.229, 0.224, 0.225])
    
    # Reverse the normalization
    tiles = [(tile * sd_resnet + mean_resnet) * 255 for tile in tiles]  # Reverse normalization and scale to [0, 255]
    tiles = [tile.astype(np.uint8) for tile in tiles]
    tiles = [cv2.cvtColor(tile, cv2.COLOR_BGR2RGB) for tile in tiles]  # Convert to RGB    
    
    # Set up the 4x4 grid for displaying the tiles
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    # Display each tile in the grid
    for i, ax in enumerate(axes.flat):
        if i < len(tiles):
            ax.imshow(tiles[i])
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

#
###

#########################
## WSI DATASET CLASSES ##
#########################


class WSIDatasetByInstance:
    """
    Dataset class for loading WSI tiles by instance, with on-demand tile loading.
    """
    def __init__(self, wsi_path_list, tile_coordinate_list, labels,
                 tile_size, level, preload_tiles, num_labels, augment=False):
        """
        Initialize the dataset.
        """
        ## Properties
        self.wsi_path_list = wsi_path_list
        self.tile_coordinate_list = tile_coordinate_list
        self.labels = labels
        self.tile_size = tile_size
        self.level = level
        self.preload_tiles = preload_tiles
        self.num_samples = len(wsi_path_list)
        self.num_labels = num_labels
        self.augment = augment

        ## Initialize lists to store instance data
        self.loaded_instance_coordinates = []
        self.loaded_labels = []
        self.loaded_wsi_paths = []
        self.loaded_sample_ids = []
        self.loaded_instances = []  # Initialize this list to avoid potential errors

        ## Preload only coordinates, labels, and paths; not the actual tiles
        self.preload_instances()

    def __len__(self):
        return len(self.loaded_instance_coordinates)

    def __getitem__(self, index):
        """
        Get a tile and its label.
        """
        if self.preload_tiles:
            instance = self.loaded_instances[index]
        else:
            tile_coordinates = [self.loaded_instance_coordinates[index]]
            wsi_path = self.loaded_wsi_paths[index]
            instance = slide_path_to_tiles_at_coordinates(wsi_path, tile_coordinates, self.tile_size, self.level)

            if len(instance) > 0:  # Ensure that the tile was successfully loaded
                instance = instance[0]
            else:
                raise ValueError(f"Tile could not be loaded from {wsi_path} at coordinates {tile_coordinates}")

            if self.augment:
                instance = augment_image(instance)

            instance = preprocess_image_resnet18(instance)

        label = self.loaded_labels[index]

        return instance, label

    def preload_instances(self):
        """
        Preload instance coordinates, labels, and WSI paths.
        """
        for i in range(self.num_samples):
            wsi_path = self.wsi_path_list[i]
            tile_coordinates = self.tile_coordinate_list[i]
            label = self.labels[i]
            num_instances = len(tile_coordinates)

            self.loaded_instance_coordinates.extend(tile_coordinates)

            labels = np.repeat(label, num_instances, axis=0)
            labels = labels.reshape((num_instances, self.num_labels))
            self.loaded_labels.extend(labels)

            wsi_paths = np.repeat(wsi_path, num_instances, axis=0)
            self.loaded_wsi_paths.extend(wsi_paths)

            sample_id = os.path.basename(wsi_path)
            sample_ids = np.repeat(sample_id, num_instances, axis=0)
            self.loaded_sample_ids.extend(sample_ids)

            if self.preload_tiles:
                tiles = slide_path_to_tiles_at_coordinates(wsi_path, tile_coordinates, self.tile_size, self.level)
                for i in range(len(tiles)):
                    if self.augment:
                        tiles[i] = augment_image(tiles[i])
                    tiles[i] = preprocess_image_resnet18(tiles[i])
                self.loaded_instances.extend(tiles)
             
#
###