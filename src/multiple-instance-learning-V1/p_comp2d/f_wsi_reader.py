#####
## ##
#####

##############
## INITIATE ##
##############

## Imports
import time
import openslide as ops
import numpy as np
import cv2

#
###

#################################
## MULTIFORMAT SLIDE FUNCTIONS ##
#################################

def read_slide(slide_path):
    """
    Open a slide image, supporting both .svs and .png formats.
    """
    if slide_path.lower().endswith('.svs'):
        slide = ops.OpenSlide(slide_path)
    elif slide_path.lower().endswith('.png'):
        slide = cv2.imread(slide_path)
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB) # WIP: to fix
    else:
        raise ValueError("Unsupported file format. Only .svs and .png are supported.")

    return slide

def get_dimensions(slide):
    """
    Get dimensions of the slide image.
    """
    if isinstance(slide, ops.OpenSlide):
        return slide.dimensions
    elif isinstance(slide, np.ndarray):
        return slide.shape[:2]
    else:
        raise ValueError("Unsupported slide type.")

def read_region(slide, location_xy, level, tile_size_xy):
    """
    Read a region from the slide.
    """
    if isinstance(slide, ops.OpenSlide):
        slide_region = np.array(slide.read_region(location_xy, level, tile_size_xy))
    elif isinstance(slide, np.ndarray):
        x, y = location_xy
        w, h = tile_size_xy
        slide_region = slide[y:y+h, x:x+w, :]
        slide_region = cv2.cvtColor(slide_region, cv2.COLOR_BGR2RGB) # WIP: to fix
    else:
        raise ValueError("Unsupported slide type.")

    return slide_region

def get_thumbnail(slide, size_x, size_y):
    """
    Get a thumbnail of the slide image.
    """
    if isinstance(slide, ops.OpenSlide):
        thumbnail = np.array(slide.get_thumbnail((size_x, size_y)))
    elif isinstance(slide, np.ndarray):
        thumbnail = cv2.resize(slide, (size_x, size_y))
    else:
        raise ValueError("Unsupported slide type.")

    return thumbnail

#
###

###############
## FUNCTIONS ##
###############

def slide_path_to_tiles_at_coordinates(svs_path, tile_coordinates, tile_size, level):
    """
    """
    ## Open slide
    slide = read_slide(svs_path)
    slide_size = get_dimensions(slide)
 
    ## Create tile stack
    tiles = []  # List to store the tiles
    for tile_coordinates_ in tile_coordinates:
        tile_x = tile_coordinates_[0]
        tile_y = tile_coordinates_[1]
        tile = read_region(slide, (tile_x, tile_y), level, (tile_size, tile_size))
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        # tile = tile/tile.max()*255
        tiles.append(tile)

    return tiles

def slide_to_tiles_at_coordinates(slide, tile_coordinates, tile_size, level):
    """
    """
    ## Create tile stack
    tiles = []  # List to store the tiles
    for tile_coordinates_ in tile_coordinates:
        tile_x = tile_coordinates_[0]
        tile_y = tile_coordinates_[1]
        tile = read_region(slide, (tile_x, tile_y), level, (tile_size, tile_size))
        # tile = np.array(slide.read_region((tile_x, tile_y), level, (tile_size, tile_size)))
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        # tile = tile/tile.max()*255
        tiles.append(tile)

    return tiles

def reassemble_tiles(tiles, coordinates, image_width=0, image_height=0):
    """
    """
    ## Determine dimensions of the original image
    tile_width = tiles[0].shape[0]
    tile_height = tiles[0].shape[1]
    max_x = max(coord[0] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)
    if image_width == 0:
        image_width = max_x + tile_width  # Assuming the width of each tile is known
    if image_height == 0:
        image_height = max_y + tile_height  # Assuming the height of each tile is known

    ## Assemble the image
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    for tile, (x, y) in zip(tiles, coordinates):
        img[y:y+tile_height, x:x+tile_width] = tile
    # img = np.mean(img, axis=2).reshape(img.shape[0], img.shape[1], 1)

    return img

def get_tile_map(slide, tile_size, downsampling_factor=30):
    """
    """
    ## Slide properties
    slide_size_level_0 = np.array(get_dimensions(slide)) # get_level_dimensions(slide, 0)
    slide_size_level_0_downsampled = np.ceil(slide_size_level_0/downsampling_factor).astype(int)

    ## Get thumbnail
    thumbnail = get_thumbnail(slide, slide_size_level_0_downsampled[0], slide_size_level_0_downsampled[1])

    ## Format thumbnail
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):
        thumbnail[:,:,i] = clahe.apply(thumbnail[:,:,i])

    ## Canvas properties
    num_tiles = np.ceil(slide_size_level_0/tile_size).astype("int") 
    canvas_size_level_0 = num_tiles * tile_size
    canvas_size_level_0_downsampled = np.ceil(canvas_size_level_0/downsampling_factor).astype(int)
    
    ## Pad
    padding = canvas_size_level_0_downsampled - slide_size_level_0_downsampled
    if padding[0] > 0:
        thumbnail = np.concatenate((thumbnail, np.zeros((thumbnail.shape[0], padding[0], thumbnail.shape[2]))), 1)
    if padding[1] > 0:
        thumbnail = np.concatenate((thumbnail, np.zeros((padding[1], thumbnail.shape[1], thumbnail.shape[2]))), 0)

    ## Format
    thumbnail = thumbnail.astype(np.uint8)
    # thumbnail = thumbnail * apply_all_filters(thumbnail)
    tile_map = cv2.resize(thumbnail, (num_tiles[0], num_tiles[1]))
    tile_map = tile_map.astype(np.uint8)

    return thumbnail, tile_map

def overlay_shifted_copies(image):
    """
    """
    ## Get paddings
    zeros_x = np.zeros((1, image.shape[1], image.shape[2]))
    zeros_y = np.zeros((image.shape[0], 1, image.shape[2]))

    ## Pad images
    image_r = np.concatenate((image[1:image.shape[0]], zeros_x), 0)
    image_l = np.concatenate((zeros_x, image[0:image.shape[0]-1]),0)
    image_u = np.concatenate((image[:, 1:image.shape[1]], zeros_y),1)
    image_d = np.concatenate((zeros_y, image[:, 0:image.shape[1]-1]),1)

    ## Combine images
    final_image = image + image_r + image_l + image_u + image_d
    
    return final_image

def apply_all_filters(img):
    """
    """
    ## Initiate
    mask = np.ones((img.shape[0], img.shape[1])).astype(np.uint8)

    ## Apply filters
    mask = mask * (np.mean(img, 2) > 50) # black
    mask = mask * (np.std(img, 2) > 10) # grey
    mask = mask * (np.mean(img, 2) < 240) # white
    mask = mask * (img[:,:,0] < img[:,:,2] + 30) # red marker
    mask = mask * (img[:,:,1] < img[:,:,2] + 20) # green marker
    mask = mask * (img[:,:,2] < img[:,:,0] + 30) # blue marker

    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    return mask

def apply_all_transforms(img):
    """
    """
    ## Diffusion 
    for k in range(3):
        img = overlay_shifted_copies(img) > 0
    img = img.astype(np.uint8)

    # ## dilation
    # kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size
    # img = cv2.dilate(img, kernel, iterations=2)
    # img = img.astype(np.uint8)

    ## Erosion
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size
    img = cv2.erode(img, kernel, iterations=1)
    img = img.astype(np.uint8)

    return img

#
###
