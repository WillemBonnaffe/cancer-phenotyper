#####################
## f_wsi_reader.py ##
#####################

## Goal:
## Define functions for reading and processing whole slide images (WSIs) in various formats,
## including extracting tiles and reassembling them.
## Author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## Imports
import time
import openslide as ops
import numpy as np
import cv2

## Import modules
from f_extract_objects import extract_object_pixels


#
###

###############
## FUNCTIONS ##
###############

## read_slide
## Goal:
## Open a slide image, supporting both .svs and .png formats.
## Inputs:
## slide_path (str): Path to the slide image file.
## Outputs:
## slide (OpenSlide object or numpy array): Slide object or image array.
def read_slide(slide_path):
    
    if slide_path.lower().endswith('.svs'):
        slide = ops.OpenSlide(slide_path)        
    elif slide_path.lower().endswith('.png'):
        slide = cv2.imread(slide_path)
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB) # opencv convention
    elif slide_path.lower().endswith('.tiff'):
        slide = cv2.imread(slide_path)
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB) # opencv convention
    else:
        raise ValueError("Unsupported file format. Only .svs or .png or .tiff are supported.")    

    return slide

## get_dimensions
## Goal:
## Get dimensions of the slide image.
## Inputs:
## slide (OpenSlide object or numpy array): Slide object or image array.
## Outputs:
## dimensions (tuple): Dimensions of the slide.
def get_dimensions(slide):
    
    if isinstance(slide, ops.OpenSlide):
        return np.flip(slide.dimensions) # (W, H) --> (H, W)
    elif isinstance(slide, np.ndarray):        
        return slide.shape[:2] # (H, W)
    else:
        raise ValueError("Unsupported slide type.")

## read_region
## Goal:
## Read a region from the slide.
## Inputs:
## slide (OpenSlide object or numpy array): Slide object or image array.
## location_xy (tuple): Coordinates of the top-left corner of the region.
## level (int): Level of the slide pyramid to read from.
## tile_size_xy (tuple): Size of the region to read (height, width).
## Outputs:
## patch (numpy array): Extracted region from the slide.
def read_region(slide, location_xy, level, tile_size_xy):

    ## Load slide and read region
    if isinstance(slide, ops.OpenSlide):
        location_yx = (location_xy[1], location_xy[0]) # openslide convention is (W, H)
        tile_size_yx = (tile_size_xy[1], tile_size_xy[0]) # openslide convention is (W, H)
        patch = np.array(slide.read_region(location_yx, level, tile_size_yx))
        patch = patch[:,:,:3] # openslide convention images have 4 channels
    elif isinstance(slide, np.ndarray):
        x, y = location_xy
        h, w = tile_size_xy
        patch = slide[x:x+h, y:y+w, :]
    else:
        raise ValueError("Unsupported slide type.")

    return patch

## slide_path_to_tiles_at_coordinates
## Goal:
## Extract tiles from a slide at specified coordinates.
## Inputs:
## svs_path (str): Path to the slide image file.
## tile_coordinates (list of tuples): List of coordinates for tile extraction.
## tile_size (int): Size of each tile (width and height).
## level (int): Level of the slide pyramid to extract tiles from.
## Outputs:
## tiles (list of numpy arrays): Extracted tiles from the slide.
def slide_path_to_tiles_at_coordinates(svs_path, tile_coordinates, tile_size, level):
    
    ## Open slide
    slide = read_slide(svs_path)
    slide_size = get_dimensions(slide)
 
    ## Create tile stack
    tiles = slide_to_tiles_at_coordinates(slide, tile_coordinates, tile_size, level)
    
    return tiles

## slide_to_tiles_at_coordinates
## Goal:
## Extract tiles from a slide object at specified coordinates.
## Inputs:
## slide (OpenSlide object or numpy array): Slide object or image array.
## tile_coordinates (list of tuples): List of coordinates for tile extraction.
## tile_size (int): Size of each tile (width and height).
## level (int): Level of the slide pyramid to extract tiles from.
## Outputs:
## tiles (list of numpy arrays): Extracted tiles from the slide.
def slide_to_tiles_at_coordinates(slide, tile_coordinates, tile_size, level):
    
    ## Create tile stack
    tiles = []  # List to store the tiles
    for tile_coordinates_ in tile_coordinates:
        tile_x, tile_y = tile_coordinates_
        tile = read_region(slide, (tile_x, tile_y), level, (tile_size, tile_size))
        # tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tiles.append(tile)

    return tiles

## get_tile_map
## Goal:
## Generate a tile map for a slide, showing the spatial arrangement of tiles.
## Inputs:
## slide (OpenSlide object or numpy array): Slide object or image array.
## tile_size (int): Size of each tile (width and height).
## downsampling_factor (int): Factor by which to downsample the slide for the map.
## Outputs:
## thumbnail (numpy array): Downsampled thumbnail of the slide.
## tile_map (numpy array): Map showing the arrangement of tiles.
def get_tile_map(slide, tile_size, downsampling_factor=16):
    
    ## Slide properties
    slide_size_level_0 = np.array(get_dimensions(slide))    

    ## Canvas properties
    num_tiles = np.ceil(slide_size_level_0 / tile_size).astype("int")
    canvas_size_level_0 = num_tiles * tile_size
    canvas_size_level_0_downsampled = canvas_size_level_0 // downsampling_factor 

    ## Initialize the thumbnail
    thumbnail_height = canvas_size_level_0_downsampled[0]
    thumbnail_width = canvas_size_level_0_downsampled[1]
    thumbnail = np.zeros((thumbnail_height, thumbnail_width, 3), dtype=np.uint8)
    
    ## Iterate through each tile and build the thumbnail
    for x in range(num_tiles[0]):
        for y in range(num_tiles[1]):

            ## Coords of tile
            x_coord = x * tile_size
            y_coord = y * tile_size

            ## Extract tile from the slide
            tile = read_region(slide, (x_coord, y_coord), 0, (tile_size, tile_size))
            # tile = slide.read_region((x_coord, y_coord), 0, (tile_size, tile_size)).convert('RGB')
            tile_np = np.array(tile)

            ## Check that the tile is not truncated
            if (tile_np.shape[0] == tile_size) & (tile_np.shape[1] == tile_size):

                ## Downsample the tile
                downsampled_tile = cv2.resize(tile_np, (tile_size // downsampling_factor, tile_size // downsampling_factor))
    
                ## Place downsampled tile in the correct position in the thumbnail
                thumb_x = x * (tile_size // downsampling_factor)
                thumb_y = y * (tile_size // downsampling_factor)
                thumbnail[thumb_x:thumb_x + downsampled_tile.shape[0],
                          thumb_y:thumb_y + downsampled_tile.shape[1]] = downsampled_tile

    ## Format the thumbnail and create the tile map
    thumbnail = thumbnail.astype(np.uint8)
    tile_map = cv2.resize(thumbnail, (num_tiles[1], num_tiles[0]))

    ## End
    tile_map = tile_map.astype(np.uint8)
    return thumbnail, tile_map, canvas_size_level_0

## get_tile_map_fast
## Goal:
## Generate a tile map for a slide, showing the spatial arrangement of tiles, but faster.
## Inputs:
## slide (OpenSlide object or numpy array): Slide object or image array.
## tile_size (int): Size of each tile (width and height).
## downsampling_factor (int): Factor by which to downsample the slide for the map.
## Outputs:
## thumbnail (numpy array): Downsampled thumbnail of the slide.
## tile_map (numpy array): Map showing the arrangement of tiles.
def get_tile_map_fast(slide, tile_size, downsampling_factor=16):
    
    ## Slide properties
    slide_size_level_0 = np.array(get_dimensions(slide))    

    ## Canvas properties
    num_tiles = np.ceil(slide_size_level_0 / tile_size).astype("int")
    canvas_size_level_0 = num_tiles * tile_size
    canvas_size_level_0_downsampled = canvas_size_level_0 // downsampling_factor 

    ## Initialize the thumbnail
    thumbnail_height = canvas_size_level_0_downsampled[0]
    thumbnail_width = canvas_size_level_0_downsampled[1]
    
    ## Get thumbnail
    if isinstance(slide, ops.OpenSlide):
        thumbnail = np.array(slide.get_thumbnail((thumbnail_width, thumbnail_height))) # openslide convention (W, H)
    elif isinstance(slide, np.array):
        thumbnail = np.zeros((thumbnail_height, thumbnail_width, 3), dtype=np.uint8)    

    ## Format the thumbnail and create the tile map
    thumbnail = thumbnail.astype(np.uint8)
    tile_map = cv2.resize(thumbnail, (num_tiles[1], num_tiles[0]))

    ## End
    tile_map = tile_map.astype(np.uint8)
    return thumbnail, tile_map, canvas_size_level_0

## reassemble_tiles
## Goal:
## Reassemble tiles into a complete image based on their coordinates.
## Inputs:
## tiles (list of numpy arrays): List of tiles to reassemble.
## coordinates (list of tuples): List of coordinates for each tile.
## image_width (int): Width of the original image (optional).
## image_height (int): Height of the original image (optional).
## Outputs:
## img (numpy array): Reassembled image.
def reassemble_tiles(tiles, coordinates, image_height=0, image_width=0):
    
    ## Determine dimensions of the original image
    tile_height = tiles[0].shape[0]
    tile_width = tiles[0].shape[1]
    image_depth = tiles[0].shape[2]
    max_x = max(coord[0] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)
    if image_height == 0:
        image_height = max_x + tile_height
    if image_width == 0:
        image_width = max_y + tile_width

    ## Assemble the image
    img = np.zeros((image_height, image_width, image_depth), dtype=np.uint8)
    for tile, (x, y) in zip(tiles, coordinates):
        if (tile.shape[0] == tile_height) & (tile.shape[1] == tile_width):
            img_ = img[x:x+tile_height, y:y+tile_width]
            if (img_.shape[0] == tile_height) & (img_.shape[1] == tile_width): # Make sure tile is not truncated
                tile = np.max(np.stack([img_, tile]), axis=0)
                img[x:x+tile_height, y:y+tile_width] = tile

    return img

## apply_all_filters
## Goal:
## Apply a series of filters to the image to mask undesired areas.
## Inputs:
## img (numpy array): Image to apply filters on.
## low (int): Low-pass filter.
## std (int): Standard deviation threshold.
## high (int): High-pass filter.
## Outputs:
## mask (numpy array): Mask highlighting areas of interest.
def apply_all_filters(img, low=50, std=10, high=230):
    
    ## Initiate
    mask = np.ones((img.shape[0], img.shape[1])).astype(np.uint8)

    ## Apply filters
    mask = mask * (np.mean(img, 2) > low)  # Black
    mask = mask * (np.std(img, 2) > std)  # Grey
    mask = mask * (np.mean(img, 2) < high)  # White
    # mask = mask * (img[:, :, 0] < img[:, :, 2] + 30)  # Red marker
    # mask = mask * (img[:, :, 1] < img[:, :, 2] + 20)  # Green marker
    # mask = mask * (img[:, :, 2] < img[:, :, 0] + 30)  # Blue marker

    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)    
    return mask

## apply_all_transforms
## Goal:
## Apply various transformations to the image, including diffusion and erosion.
## Inputs:
## img (numpy array): Image to apply transformations on.
## Outputs:
## img (numpy array): Transformed image.
def apply_all_transforms(img):
    
    ## Diffusion 
    for k in range(3):
        img = overlay_shifted_copies(img) > 0
    img = img.astype(np.uint8)

    ## Erosion
    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size if needed
    img = cv2.erode(img, kernel, iterations=1)
    img = img.astype(np.uint8)

    img = img.reshape(img.shape[0], img.shape[1], 1)
    return img

## overlay_shifted_copies
## Goal:
## Overlay shifted copies of the image to create a combined effect.
## Inputs:
## image (numpy array): Image to overlay with shifted copies.
## Outputs:
## final_image (numpy array): Image after overlaying shifted copies.
def overlay_shifted_copies(image):
    
    ## Get paddings
    zeros_x = np.zeros((1, image.shape[1], image.shape[2]))
    zeros_y = np.zeros((image.shape[0], 1, image.shape[2]))

    ## Pad images
    image_r = np.concatenate((image[1:image.shape[0]], zeros_x), 0)
    image_l = np.concatenate((zeros_x, image[0:image.shape[0]-1]), 0)
    image_u = np.concatenate((image[:, 1:image.shape[1]], zeros_y), 1)
    image_d = np.concatenate((zeros_y, image[:, 0:image.shape[1]-1]), 1)

    ## Combine images
    final_image = image + image_r + image_l + image_u + image_d
    
    return final_image

## select_patches_on_grid
## Goal:
## Select a subset of coordinates that are closest to regular grid points.
## Inputs:
## coordinates (list): List of patch coordinates in the form [[x1, y1], [x2, y2], ...].
## grid_spacing (int): The spacing between grid points.
## Outputs:
## selected_coordinates (list): List of coordinates selected based on grid points.
def select_patches_on_grid(coordinates, grid_spacing):
    
    ## Convert coordinates to a numpy array
    coordinates = np.array(coordinates)

    ## Determine the min and max range of coordinates to define the grid
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)

    ## Generate grid points
    grid_x = np.arange(min_x, max_x + grid_spacing, grid_spacing)
    grid_y = np.arange(min_y, max_y + grid_spacing, grid_spacing)
    grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)

    ## List to store selected coordinates
    selected_coordinates = []

    ## Find the nearest coordinate to each grid point
    for grid_point in grid_points:
        distances = np.linalg.norm(coordinates - grid_point, axis=1)  # Calculate Euclidean distance
        nearest_index = np.argmin(distances)  # Get the index of the nearest coordinate
        nearest_coordinate = coordinates[nearest_index]

        ## Add the coordinate to the selected list if not already included
        if list(nearest_coordinate) not in selected_coordinates:
            selected_coordinates.append(list(nearest_coordinate))

    return selected_coordinates

## get_tissue_mask
## Goal:
## Generate a mask that identifies tissue areas in a tile map, removing background regions.
## Inputs:
## tile_map (numpy array): Image from which to extract tissue areas.
## filter_low (float): Low threshold value for filtering.
## filter_std (float): Standard deviation threshold for filtering.
## filter_high (float): High threshold value for filtering.
## threshold (int): Minimum size of background objects
## Outputs:
## mask (numpy array): Mask highlighting tissue areas while removing background.
def get_tissue_mask(tile_map, filter_low, filter_std, filter_high, threshold=1000000):

    ## Extract coordinates of pixels in tile_map that contain tissue
    mask_tissue = apply_all_filters(tile_map, low=filter_low, std=filter_std, high=filter_high)
    mask_tissue = apply_all_transforms(mask_tissue)
    
    ## Get background
    objects = extract_object_pixels(1-mask_tissue)
    
    ## Remove background
    if objects != []:

        ## Select objects larger than threshold
        s = np.argwhere(np.array([len(object) for object in objects]) > threshold).flatten()
        objects_new = []
        for s_ in s:
            objects_new += [objects[s_]]
    
        if objects_new != []:
            background_tiles_coordinates = np.concatenate(objects_new)
        else: # Else select largest object
            background_tiles_coordinates = np.array(objects[np.argmax([len(object) for object in objects])])
            
        ## Get bacgkround mask        
        mask_background = np.zeros((mask_tissue.shape[0], mask_tissue.shape[1], 1))
        mask_background[background_tiles_coordinates[:,0], background_tiles_coordinates[:,1]] = 1
        
        ## Define not background mask
        mask = 1 - mask_background
        
        ## Erosion to reduce boundary
        kernel = np.ones((5, 5), np.uint8) # Adjust the kernel size if needed
        mask = cv2.erode(mask, kernel, iterations=1)
        
        ## Format
        mask = mask.astype(np.uint8)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

        ## Get largest bit of tissue
        objects = extract_object_pixels(mask)
        tissue_tiles_coordinates = np.array(objects[np.argmax([len(object) for object in objects])])
        mask = np.zeros((mask_tissue.shape[0], mask_tissue.shape[1], 1))
        mask[tissue_tiles_coordinates[:,0], tissue_tiles_coordinates[:,1]] = 1

        ## Format
        mask = mask.astype(np.uint8)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    
    else:
    
        ## Otherwise use tissue mask
        mask = mask_tissue

    return mask
    
#
###
