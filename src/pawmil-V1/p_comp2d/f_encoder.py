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

#
###

#####################
## ENCODER CLASSES ##
#####################

class BasicResNet18Encoder(nn.Module):
    """
    """
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
