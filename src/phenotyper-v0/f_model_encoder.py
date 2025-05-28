#####################
## ENCODER CLASSES ##
#####################

## Goal: Define a ResNet18 model class to generate embeddings from images.
## Author: Willem Bonnaffé (w.bonnaffe@gmail.com)

######################################
## IMPORTS AND LIBRARY DEPENDENCIES ##
######################################

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights

#
###

#######################
## CLASS DEFINITIONS ##
#######################

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=512):
        super(ResNet18FeatureExtractor, self).__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # [B, 512, 1, 1]

        self.flatten = nn.Flatten()
        self.project = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.project(x)
        return x

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=512):
        super(ResNet50FeatureExtractor, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # [B, 2048, 1, 1]

        self.flatten = nn.Flatten()
        self.project = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.project(x)
        return x

class ViTFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=512):
        super(ViTFeatureExtractor, self).__init__()

        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None)

        self.backbone.heads = nn.Identity()  # Remove classification head

        self.project = nn.Linear(768, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)  # [B, 768]
        x = self.project(x)   # [B, 512]
        return x

#
###

"""
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
"""

#
###