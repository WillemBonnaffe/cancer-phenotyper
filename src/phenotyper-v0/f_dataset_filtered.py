##############
## INITIATE ##
##############

## Imports
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image # See if results are robust with using opencv pre-processing

## Import modules
from f_model_unet import UNet
from f_model_transunet import UNetTransformer

#
###

###############
## FUNCTIONS ##
###############

def pad_image_even(img):
    """    
    ## Goal: Pad image to an even size.
    ## Inputs: 
    ## - img (numpy array): Image to pad.
    ## Outputs:
    ## - img_padded (numpy array): Image padded.
    """
    pad_size = np.max(img.shape)
    if img.shape[0] < pad_size:
        img = np.concatenate((img, np.zeros((pad_size - img.shape[0], img.shape[1], img.shape[2]))), axis=0)
    if img.shape[1] < pad_size:
        img = np.concatenate((img, np.zeros((img.shape[0], pad_size - img.shape[1], img.shape[2]))), axis=1)
    return img.astype(int)

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

def image_to_tensor(image):
    """
    ## Goal: Format image to tensor.
    ## Inputs: 
    ## - image (numpy array): Image as a numpy array.
    ## Outputs:
    ## - tensor (torch tensor): Formatted tensor.    
    """
    image = pad_image_even(image)
    image = cv2.resize(image.astype('uint8'), (1024, 1024), interpolation=cv2.INTER_LINEAR)    
    tensor = torch.tensor(image.swapaxes(2, 0).swapaxes(1, 2)).float()    
    return tensor
    
def tensor_to_tensor(image):
    """
    ## Goal: Format image to tensor.
    ## Inputs: 
    ## - image (numpy array): Image as a numpy array.
    ## Outputs:
    ## - tensor (torch tensor): Formatted tensor.    
    """
    transform = transforms.Compose([
        # transforms.ToTensor(), # Converts to tensor and scales to [0, 1]
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    tensor = transform(image)
    return tensor

def filter_tensor(tensor, filter_model):
    """
    ## Goal: Filter tensor using a segmentation model.
    ## Inputs:
    ## - tensor (torch tensor): Formatted tensor.
    ## - filter_model (torch model): Model to filter tensor.
    ## Outputs:
    ## - tensor_filtered (torch tensory): Filtered tensor.
    """
    tensor = tensor[None, :, :, :] # To add batch 
    tensor_filtered = filter_model(tensor)
    tensor_filtered = tensor_filtered[0] # To remove batch
    return tensor_filtered

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

class FilteredImageDataset(Dataset):
    """    
    ## Goal: Load images from a folder, apply preprocessing and augmentations, and return image identifiers.
    ## Inputs:
    ## - folder_path (str): Path to the folder containing images.
    ## - augment (bool): Flag to apply data augmentations.
    ## Outputs:
    ## - Image tensor with applied preprocessing and augmentations, and its identifier.
    """    
    def __init__(self, folder_path, augment=False, device='mps'):
        
        ## Initialization
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]        
        self.augment = augment
        self.device = device

        ## Load filter model        
        model = UNetTransformer(n_classes=3).to(device)
        state_dict = torch.load('models/model_transunet_V1_1.pth', map_location=device)
        model.load_state_dict(state_dict) # Load weights      
        self.filter_model = model

        ## Define methods
        self.preprocess = preprocess_image        
        self.format_filter = image_to_tensor
        self.filter = filter_tensor
        self.format_encoder = tensor_to_tensor
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
        image = self.format_filter(image)                
        image = image.to(self.device)

        ## Apply filter
        image = self.filter(image, self.filter_model)

        ## Apply formatting (resize, convert to tensor and normalize)
        image = self.format_encoder(image)

        ## Apply augmentations if augment flag is set
        if self.augment:
            image = self.transform(image)

        ## Return the image and its identifier (filename)
        return image, self.image_files[idx]
        
#
###