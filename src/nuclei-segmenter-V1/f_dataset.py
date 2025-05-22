##################
## f_dataset.py ##
##################

## goal: module for data loading and formatting

## author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## imports
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image 
import torchvision.transforms as T
import torch.nn.functional as F

#
###

###############
## FUNCTIONS ##
###############

## canvas embedding
## goal: embedd image in black square canvas
## img      - numpy array - image to embedd
## imgSize  - int         - size of image
## tileSize - int         - number of tiles to fit in the canvas
def embedd(img, imgSize, tileSize):
    nTiles = (np.ceil(imgSize/tileSize)).astype("int")
    canvasSize = nTiles*tileSize
    if (canvasSize - img.shape[0] > 0):
        img = np.concatenate((img,np.zeros((canvasSize - img.shape[0],img.shape[1],img.shape[2]))),axis=0)
    if (canvasSize - img.shape[1] > 0):
        img = np.concatenate((img,np.zeros((img.shape[0],canvasSize - img.shape[1],img.shape[2]))),axis=1)
    return img

## binarise_channels
## goal: binarise channels of image according to a threshold array
## img        - np.array - image to split the channels of
## thresholds - np.array - threshold according to which channels in image should be splitted
def binarise_channels(img, thresholds):
    img_new = np.copy(img)
    for i in range(len(thresholds)-1):
        img_new[:,:,i] = (img[:,:,i] >= thresholds[i]) & (img[:,:,i] < thresholds[i+1])
    return img_new

## labelise_channels
## goal: labelise channels of image according to a threshold array
## img        - np.array - image to split the channels of
## thresholds - np.array - threshold according to which channels in image should be splitted
def labelise_channels(img, thresholds):
    img_new = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for i in range(len(thresholds)-1):
        img_new[:,:,i] = ((img[:,:,i] >= thresholds[i]) & (img[:,:,i] < thresholds[i+1])) * i
    img_new = np.sum(img_new, axis = 2).reshape(img.shape[0],img.shape[1],1) # format
    return img_new

## cropflip_transform 
## goal: transform a given tensor by cropping and flipping
## tens        - tensor - tensor to transform
## alpha       - float  - scaling factor
## target_size - int    - size of the final tensor
## flip_h      - bool   - whether to flip the image horizontally
## flip_v      - bool   - whether to flip the image vertically
def cropflip_transform(tens, alpha, target_size, flip_h, flip_v):

    ## determine crop size
    crop_size = int(tens.shape[1]*alpha)

    ## if crop larger than image -> pad
    if crop_size > tens.shape[1]:
        pad_size = int((crop_size - tens.shape[1])/2)
        tens_ = F.pad(tens, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        tens_ = F.pad(tens_, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    else: # else crop
        tens_ = T.CenterCrop(crop_size)(tens)

    ## resize image
    tens_ = T.Resize(target_size)(tens_)

    ## flip
    if flip_h == True:
        tens_ = T.functional.hflip(tens_)
    if flip_v == True:
        tens_ = T.functional.vflip(tens_)

    ## terminate
    return tens_

## shuffle_channels
## goal: shuffle channels of tensor
## tens - tensor - tensor to transform
def shuffle_channels(tens):
    channels_order = np.arange(tens.shape[0])
    random.shuffle(channels_order)
    tens = tens[channels_order]
    return tens

## power_transform
## goal: power transform tensor within [0, 255]
## tens     - tensor - tensor to transform
## strength - float  - float controlling magnitude of power transform 
def power_transform(tens, strength=0.5):
    for i in range(tens.shape[0]):
        tens[i] = torch.pow(tens[i],1 - torch.rand(1) * strength)
    return tens

## brightness_transform
## goal: brightness transform tensor within [0, 255]
## tens     - tensor - tensor to transform
## strength - float  - float controlling magnitude of transform 
def brightness_transform(tens, strength=0.5):
    tens = tens*(1 - torch.rand(1) * strength)
    return tens

## labels_to_onehot
## goal: convert image with label value of segmentation classes to one hot image with as many channels as classes
## img - np.array - image containing predicted segmentation classes
## n_classes - int - number of classes
def labels_to_onehot(img, n_classes):
    img_new = np.zeros((img.shape[0], img.shape[1], n_classes))
    for i in range(n_classes):
        img_new[:,:,i] = ((img == i)*1).squeeze()
    return img_new

## multiple_labels_to_onehot
## goal: convert multiple images with label value of segmentation classes to one hot image with as many channels as classes
## images - np.array - array containing multiple images
## n_classes - int - number of semantic classes
## drop_background - bool - whether to remove background class (0)
def multiple_labels_to_onehot(images, n_classes, drop_background = False):
    labels_new = []
    for image in images:
        if drop_background == True:
            label_new_ = labels_to_onehot(image, n_classes)[:,:,1:n_classes]
            labels_new.append(label_new_.reshape(1, label_new_.shape[0], label_new_.shape[1], label_new_.shape[2]))
        else: 
            label_new_ = labels_to_onehot(image, n_classes)
            labels_new.append(label_new_.reshape(1, label_new_.shape[0], label_new_.shape[1], label_new_.shape[2]))
    labels_new = np.concatenate(labels_new, 0)
    return labels_new

#
###

#############
## CLASSES ##
#############

## dataset
## goal: class that handles the loading and formatting of segmentation training data
## path_to_datast - string - path to dataset for training containing masks and input images
## tileSize - int - size of an individual tile in pixels
## imgSize - int - size of an individual image in pixels
## thresholds - np.array - thresholds to break down the target into class labels
## n_repeat - np.array - number of times to augment the dataset
## n_skip_val - int - number of values to skip in train set and use for validation
class dataset:

    def __init__(self, path_to_dataset, tileSize, imgSize, thresholds, n_repeat, n_skip_val, batchSize):

        ## load data
        X   = [] 
        Y   = [] 
        pti = path_to_dataset
        ptd = os.listdir(path_to_dataset)
        for ptd_ in ptd:
        
            ## update
            print(ptd_)
        
            ## read tiles
            tiles = np.array(os.listdir(pti + ptd_ + "/reference/"))
            for tile in tiles:
                # print(tile)
                img = cv2.imread(pti + ptd_ + "/reference/" + tile)
                img = cv2.resize(img,(1024,1024),interpolation=cv2.INTER_LINEAR)
                img = embedd(img,imgSize,tileSize)
                X.append(img)
        
            ## read masks 
            masks = np.array(os.listdir(pti + ptd_ + "/mask/"))
            for mask in masks:
                img = cv2.imread(pti + ptd_ + "/mask/" + mask)
                img = cv2.resize(img,(1024,1024),interpolation=cv2.INTER_LINEAR)
                img = labelise_channels(img,thresholds)
                img = embedd(img,imgSize,tileSize)
                Y.append(img)
        
        ## format data
        X = np.stack(X)
        Y = np.stack(Y)
        #
        # ## DEBUG >>
        # print(Y.shape)
        # print(np.unique(Y))
        # plt.imshow(Y[0])
        # plt.savefig("tmp.png")
        # plt.close
        # ## DEBUG <<
        #
        ## format for pytorch
        X = X.swapaxes(3,2).swapaxes(2,1)
        Y = Y.swapaxes(3,2).swapaxes(2,1)

        ## DEBUG >>
        print(X.shape)
        print(Y.shape)
        ## DEBUG <<
       
        ## split in learning and testing sets
        s_l = np.arange(0,len(X))
        s_t = np.arange(0,len(X),n_skip_val).flatten()
        s_l = s_l[np.invert(np.isin(s_l,s_t))].flatten()
        #
        X_l = X[s_l]
        X_t = X[s_t]
        #
        Y_l = Y[s_l]
        Y_t = Y[s_t]
        
        ## convert to tensor
        X_l = torch.Tensor(X_l)
        X_t = torch.Tensor(X_t)
        Y_l = torch.Tensor(Y_l)
        Y_t = torch.Tensor(Y_t)

        ## augment training set
        X_l = X_l.repeat(n_repeat,1,1,1) 
        Y_l = Y_l.repeat(n_repeat,1,1,1) 
        X_t = X_t.repeat(1,1,1,1)
        Y_t = Y_t.repeat(1,1,1,1) 

        ## data augmentation
        for i in range(0,X_l.shape[0]):

            ## tranform parameters
            alpha  = np.random.uniform(0.1,3.0)
            flip_h = int(np.round(np.random.uniform(0,1,1))) == 1
            flip_v = int(np.round(np.random.uniform(0,1,1))) == 1

            ## apply transform
            X_l[i] = cropflip_transform(X_l[i], alpha = alpha, target_size = imgSize, flip_h = flip_h, flip_v = flip_v) 
            Y_l[i] = cropflip_transform(Y_l[i], alpha = alpha, target_size = imgSize, flip_h = flip_h, flip_v = flip_v) 

            ## colour transforms
            # X_l[i] = shuffle_channels(X_l[i]) # optional
            X_l[i] = power_transform(X_l[i], strength=0.1)
            X_l[i] = brightness_transform(X_l[i], strength=0.1)

        # ## format single dimensions in target
        # Y_l = Y_l.squeeze().type(torch.LongTensor) # only for cross entropy loss
        # Y_t = Y_t.squeeze().type(torch.LongTensor) # only for cross entropy loss

        ## push into dataloaders
        data_l = TensorDataset(Y_l, X_l)
        data_t = TensorDataset(Y_t, X_t)
        self.dataloader_l = DataLoader(data_l, batch_size=batchSize, shuffle=True)
        self.dataloader_t = DataLoader(data_t, batch_size=batchSize)

#
###
