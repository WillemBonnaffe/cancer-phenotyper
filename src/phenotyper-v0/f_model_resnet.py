######################################
## RESNET18 FEATURE EXTRACTOR CLASS ##
######################################

## Goal: Define a ResNet18 model class to generate embeddings from images.
## Author: Willem Bonnaffé (w.bonnaffe@gmail.com)

######################################
## IMPORTS AND LIBRARY DEPENDENCIES ##
######################################

import torch
import torch.nn as nn
from torchvision import models

#
###

######################################
## RESNET18 FEATURE EXTRACTOR CLASS ##
######################################

## ResNet18FeatureExtractor
## Goal:
## Implement a feature extractor using a pre-trained ResNet18 model to generate embeddings from images.
## Inputs:
## pretrained (bool): If True, loads the ImageNet pre-trained weights.
## Outputs:
## Embedding tensor for each input image.
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18FeatureExtractor, self).__init__()
        
        ## Load pre-trained ResNet18
        self.resnet18 = models.resnet18(pretrained=pretrained)
        
        ## Remove the final fully connected layer
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        
        ## Flatten layer to ensure output is a feature vector
        self.flatten = nn.Flatten()

    def forward(self, x):
        ## Forward pass through the ResNet18 backbone
        x = self.resnet18(x)
        
        ## Flatten the output to obtain the embedding vector
        x = self.flatten(x)
        
        return x

#
###