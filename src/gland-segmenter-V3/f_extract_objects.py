##########################
## f_extract_objects.py ##
##########################

## Goal: Define functions for extracting objects from masks.
## Author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## Imports
import numpy as np
import cv2

#
###

###############
## FUNCTIONS ##
###############

## polish
## Goal:
## Apply erosion to the image to polish the objects.
## Inputs:
## img (numpy array): Image to apply erosion on.
## Outputs:
## img (numpy array): Eroded image.
def polish(img):
    
    ## Erosion
    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size if needed
    img = cv2.erode(img, kernel, iterations=1)
    img = img.astype(np.uint8)
    
    return img

## extract_object_pixels
## Goal:
## Extract the pixel coordinates of objects from a binary segmentation mask.
## Inputs:
## mask (numpy.ndarray): A 2D binary segmentation mask where objects have a value of 1 (or 255) and background is 0.
## Outputs:
## object_pixels (list): A list of numpy arrays, each containing the (row, column) coordinates of the pixels of each object.
def extract_object_pixels(mask):
    
    ## Ensure the mask is binary (values are either 0 or 1)
    if mask.max() > 1:
        mask = mask // 255  # Convert 255 values to 1 for consistency if it's an 8-bit mask.

    ## Find connected components (objects) in the mask
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    ## Extract pixel coordinates for each object, ignoring the background (label 0)
    object_pixels = []
    for label in range(1, num_labels):
        ## Find pixel coordinates where the label is present
        pixels = np.column_stack(np.where(labels == label))
        object_pixels.append(pixels)

    return object_pixels

## get_bounding_box
## Goal:
## Get the bounding box of an object from its pixel coordinates.
## Inputs:
## coords (numpy.ndarray): Array of (row, column) coordinates of the object's pixels.
## Outputs:
## bounding_box (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max).
def get_bounding_box(coords):
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max

## get_bounding_box_square
## Goal:
## Get the bounding box of an object from its pixel coordinates, making it square.
## Inputs:
## coords (numpy.ndarray): Array of (row, column) coordinates of the object's pixels.
## Outputs:
## bounding_box (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max).
def get_bounding_box_square(coords):

    ## Min and max of coordinates
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    ## Calculate width and height
    height = x_max - x_min
    width = y_max - y_min
    
    ## Make the bounding box square by taking the larger of the two dimensions
    side_length = max(height, width)
    
    ## Center the square box around the object's bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    x_min = x_center - side_length // 2
    x_max = x_center + side_length // 2
    y_min = y_center - side_length // 2
    y_max = y_center + side_length // 2

    return x_min, y_min, x_max, y_max

## inlay_objects
## Goal:
## Visualise individual objects in an image.
## Inputs:
## img (numpy array): Image as a numpy array.
## coordinates (list): List of coordinates of pixels of objects.
## Outputs:
## img (numpy array): Image with objects inlaid in different colors.
def inlay_objects(img, coordinates):
    img = img * 0
    if len(img.shape) < 3:
        img = img[:, :, None]
    if img.shape[2] == 1: 
        img = np.repeat(img, 3, axis=2)
    for i in range(len(coordinates)):
        coordinates_i = coordinates[i].astype("int")
        color_vect = np.random.uniform(0, 1, 3) * 255
        img[coordinates_i[:, 0], coordinates_i[:, 1], 0] = color_vect[0]
        img[coordinates_i[:, 0], coordinates_i[:, 1], 1] = color_vect[1]
        img[coordinates_i[:, 0], coordinates_i[:, 1], 2] = color_vect[2]
    return img

#
###
