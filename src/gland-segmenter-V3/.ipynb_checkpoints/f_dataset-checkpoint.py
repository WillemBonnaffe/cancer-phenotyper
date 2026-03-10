######################
## f_dataset_v2.py  ##
######################

## Goal: Module for data loading, formatting, and on-the-fly augmentation using data generators.
## Author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## Imports
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import random

#
###

###############
## FUNCTIONS ##
###############

## pad_image_even
## Goal:
## Pad image to an even size.
## Inputs: 
## img (numpy array): Image to pad.
## Outputs:
## img_padded (numpy array): Image padded.
def pad_image_even(img):
    pad_size = np.max(img.shape)
    if img.shape[0] < pad_size:
        img = np.concatenate((img, np.zeros((pad_size - img.shape[0], img.shape[1], img.shape[2]))), axis=0)
    if img.shape[1] < pad_size:
        img = np.concatenate((img, np.zeros((img.shape[0], pad_size - img.shape[1], img.shape[2]))), axis=1)
    return img.astype(int)

## labelise_channels
## Goal: 
## Labelise channels of image according to a threshold array.
## Inputs:
## img (numpy array): Image to split the channels of.
## thresholds (numpy array): Thresholds according to which channels in the image should be split.
## Outputs:
## img_new (numpy array): Image with labelised channels.
def __OLD__labelise_channels(img, thresholds):
    img_new = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for i in range(len(thresholds) - 1):
        img_new[:, :, i] = ((img[:, :, i] >= thresholds[i]) & (img[:, :, i] < thresholds[i + 1])) * i
    img_new = np.sum(img_new, axis=2).reshape(img.shape[0], img.shape[1], 1) # Format
    return img_new

## labelise_channels
## Goal: 
## Labelise channels of image according to a threshold array.
## Inputs:
## img (numpy array): Image to split the channels of.
## thresholds (numpy array): Thresholds according to which channels in the image should be split.
## Outputs:
## img_new (numpy array): Image with labelised channels.
def labelise_channels(img, thresholds):
    img_new = np.zeros((img.shape[0], img.shape[1], len(thresholds)-1))
    img = img.mean(axis=2)
    for i in range(len(thresholds) - 1):
        img_new[:, :, i] = ((img >= thresholds[i]) & (img < thresholds[i + 1])) * i
    img_new = np.sum(img_new, axis=2).reshape(img.shape[0], img.shape[1], 1) # Format
    return img_new

## cropflip_transform
## Goal: 
## Transform a given tensor by cropping and flipping.
## Inputs:
## tens (tensor): Tensor to transform.
## alpha (float): Scaling factor.
## target_size (int): Size of the final tensor.
## flip_h (bool): Whether to flip the image horizontally.
## flip_v (bool): Whether to flip the image vertically.
## Outputs:
## tens_ (tensor): Transformed tensor.
def cropflip_transform(tens, alpha, target_size, flip_h, flip_v):

    ## Determine crop size
    crop_size = int(tens.shape[1]*alpha)

    ## If crop larger than image -> pad
    if crop_size > tens.shape[1]:
        pad_size = int((crop_size - tens.shape[1])/2)
        tens_ = F.pad(tens, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        tens_ = F.pad(tens_, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    else: # Else crop
        tens_ = T.CenterCrop(crop_size)(tens)

    ## Resize image
    tens_ = T.Resize(target_size, antialias=None)(tens_)

    ## Flip
    if flip_h==True:
        tens_ = T.functional.hflip(tens_)
    if flip_v==True:
        tens_ = T.functional.vflip(tens_)

    return tens_

## power_transform
## Goal: 
## Apply power transform to tensor within [0, 255].
## Inputs:
## tens (tensor): Tensor to transform.
## strength (float): Float controlling magnitude of power transform.
## Outputs:
## tens (tensor): Transformed tensor.
def power_transform(tens, strength=0.5):
    for i in range(tens.shape[0]):
        tens[i] = torch.pow(tens[i], 1 - torch.rand(1) * strength)
    return tens

## brightness_transform
## Goal: 
## Apply brightness transform to tensor within [0, 255].
## Inputs:
## tens (tensor): Tensor to transform.
## strength (float): Float controlling magnitude of transform.
## Outputs:
## tens (tensor): Transformed tensor.
def brightness_transform(tens, strength=0.5):
    tens = tens*(1 - torch.rand(1) * strength)
    return tens

## augment
## Goal:
## Apply on-the-fly augmentations such as crop, flip, brightness, and power transform.
## Inputs:
## img (tensor): Image tensor to augment.
## mask (tensor): Mask tensor to augment.
## Outputs:
## img (tensor): Augmented image tensor.
## mask (tensor): Augmented mask tensor.
def augment(img, mask, alpha_min=0.1, alpha_max=1.25, power_strength=0.05, brigtness_strength=0.05):    
    
    ## Transform parameters
    alpha = np.random.uniform(alpha_min, alpha_max)
    flip_h = int(np.round(np.random.uniform(0, 1, 1))) == 1
    flip_v = int(np.round(np.random.uniform(0, 1, 1))) == 1

    ## Apply transform
    img = cropflip_transform(img, alpha=alpha, target_size=img.shape[1], flip_h=flip_h, flip_v=flip_v) 
    mask = cropflip_transform(mask, alpha=alpha, target_size=mask.shape[1], flip_h=flip_h, flip_v=flip_v) 

    ## Swap B and R channels
    if random.choice([True, False]):
        img = img[[2, 1, 0], :, :] # Swapping the 0th (B) and 2nd (R) channels
 
    ## Colour transforms    
    img = power_transform(img, strength=power_strength)
    img = brightness_transform(img, strength=brigtness_strength)

    ## End
    return img, mask

#
###

#############
## CLASSES ##
#############

## SegmentationDataset
## Goal: 
## Class that handles lazy loading, formatting, and on-the-fly augmentation of segmentation data.
## Inputs:
## path_to_dataset (str): Path to dataset for training containing masks and input images.
## thresholds (numpy array): Thresholds to break down the target into class labels.
## transform (function): Augmentation function to apply on-the-fly transformations.
class SegmentationDataset(Dataset):

    ## Initialize dataset paths and thresholds
    def __init__(self, path_to_dataset, thresholds, transform=None):
        self.path_to_dataset = path_to_dataset
        self.thresholds = thresholds
        self.transform = transform

        ## Collect image and mask paths
        self.image_paths = []
        self.mask_paths = []
        for folder in os.listdir(path_to_dataset):
            if folder != ".DS_Store":
                image_folder = os.path.join(path_to_dataset, folder, "reference")
                mask_folder = os.path.join(path_to_dataset, folder, "mask")
                    
                for image_file in os.listdir(image_folder):                    
                    if image_file != ".DS_Store":                        
                        self.image_paths.append(os.path.join(image_folder, image_file))
                        self.mask_paths.append(os.path.join(mask_folder, image_file.replace('_img',''))) # Comment out if mask file name identical to reference

                """
                ## Avail if mask file names identical to reference file names
                for mask_file in os.listdir(mask_folder):
                    if mask_file != ".DS_Store":                        
                        self.mask_paths.append(os.path.join(mask_folder, mask_file))        
                """

    ## Return length of dataset
    def __len__(self):
        return len(self.image_paths)

    ## Load and process a single sample
    def __getitem__(self, idx):
        
        ## Load image and mask
        img = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx])

        ## Pad and resize images
        img = pad_image_even(img)
        mask = pad_image_even(mask)        
        img = cv2.resize(img.astype('uint8'), (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype('uint8'), (1024, 1024), interpolation=cv2.INTER_LINEAR)

        ## Labelise channels in mask (convert brightness levels to classes)
        mask = labelise_channels(mask, self.thresholds) # + 1 # As background is 0

        ## Convert to PyTorch tensors
        img = torch.tensor(img.swapaxes(2, 0).swapaxes(1, 2)).float()
        mask = torch.tensor(mask.swapaxes(2, 0).swapaxes(1, 2)).float()        

        ## Apply augmentations if provided
        if self.transform:
            img, mask = self.transform(img, mask)

        ## Format single dimensions in target
        mask = mask.squeeze().type(torch.LongTensor) # Only for cross entropy loss    
        
        return img, mask

#
###
