##############
## INITIATE ##
##############

## Imports
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image # See if results are robust with using opencv pre-processing

#
###

###############
## FUNCTIONS ##
###############

def preprocess_image(image):
    """
    ## Goal: Preprocess image.
    ## Inputs: 
    ## - image (numpy array): Image as a numpy array.
    ## Outputs:
    ## - Preprocessed image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(image)
    l = clahe.apply(l)
    image = cv2.merge((l, a, b))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)    
    return image

def image_to_tensor(image, resize):
    """
    ## Goal: Format image to tensor.
    ## Inputs: 
    ## - image (numpy array): Image as a numpy array.
    ## - resize (int): Size of resized image in pixels.
    ## Outputs:
    ## - tensor (torch tensor): Formatted tensor.    
    """
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to tensor and scales to [0, 1]
        transforms.Resize((resize, resize)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    tensor = transform(image)
    return tensor

def transform_tensor(tensor):
    """
    ## Goal: transform image or tensor.
    ## Inputs: 
    ## - tensor (torch tensor): Formatted tensor.    
    ## Outputs:
    ## - tensor (torch tensor): Transformed tensor. 
    """
    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),            
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4, hue=0.4)
    ])
    tensor_transformed = transformations(tensor)
    return tensor_transformed
    
#
###

##########################
## CUSTOM DATASET CLASS ##
##########################

class ImageDataset(Dataset):
    """
    ## ProstateMicroscopyDataset
    ## Goal:
    ## Load images from a folder, apply preprocessing and augmentations, and return image identifiers.
    ## Inputs:
    ## folder_path (str): Path to the folder containing images.
    ## augment (bool): Flag to apply data augmentations.
    ## Outputs:
    ## Image tensor with applied preprocessing and augmentations, and its identifier.
    """    
    def __init__(self, folder_path, augment=False, resize=224, file_format='.png'):
        
        ## Initialization
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(file_format)]
        # self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') & (f.endswith('_both.png')==False) & (f.endswith('_glands.png')==False) & (f.endswith('_nuclei.png')==False)]        
        self.augment = augment
        self.resize = resize

        ## Define methods
        self.preprocess = preprocess_image        
        self.format = image_to_tensor
        self.transform = transform_tensor          

    def __len__(self):         
        return len(self.image_files) # Return the total number of images

    def __getitem__(self, idx):
        
        ## Load image
        img_name = os.path.join(self.folder_path, self.image_files[idx])        
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## Apply preprocessing: CLAHE, resize, normalization
        image = self.preprocess(image)
        
        ## Apply formatting (resize, convert to tensor and normalize)
        image = self.format(image, self.resize)

        ## Apply augmentations if augment flag is set
        if self.augment:
            image = self.transform(image)

        ## Return the image and its identifier (filename)
        return image, self.image_files[idx]
        
#
###