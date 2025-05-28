#####
## ##
#####

##############
## INITIATE ##
##############

## Imports
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights

#
###

#####################
## ENCODER CLASSES ##
#####################

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

    def get_layers(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.project(x)
        return [x, x, x]


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

    def get_layers(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.project(x)
        return [x, x, x]

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

    def get_layers(self, x):
        x = self.backbone(x)        
        x = self.project(x)
        return [x, x, x]

"""
class BasicResNet18Encoder(nn.Module):
    def __init__(self):
        super(BasicResNet18Encoder, self).__init__()

        ## Backbone
        self.backbone = models.resnet18(pretrained=True)
        # self.backbone.fc = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def get_layers(self, x):
        x = self.backbone(x)
        return [x, x, x]
"""

# class BasicResNet18Encoder(nn.Module):
#     """
#     """
#     def __init__(self, n_features=128, n_output=1, n_hidden=256):
#         super(BasicResNet18Encoder, self).__init__()
# 
#         ## backbone
#         self.backbone = models.resnet18(pretrained=True)
#         self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_features)
#         #
#         ## classifier
#         self.pl = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))
#         self.link = nn.Sigmoid()
#         # self.link = nn.Identity()
# 
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.pl(x)
#         # x = self.link(x)
#         return x
# 
#     def get_layers(self, x):
#         x = self.backbone(x); e = x # save embeddings
#         x = self.pl(x); logit = x
#         x = self.link(x); p = x # save predictions
#         return [p, e, logit]

#
###
