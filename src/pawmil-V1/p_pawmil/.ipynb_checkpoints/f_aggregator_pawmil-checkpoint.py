#####
## ##
#####

## Goal:

##############
## INITIATE ##
##############

## Imports
import torch
import torch.nn as nn
# import torchvision.models as models
# import torch.nn.functional as F
# import torchvision.transforms as transforms

#
###

########################
## AGGREGATOR CLASSES ##
########################

class AttentionFirstAggregator(nn.Module):
    """
    """
    def __init__(self, n_features, n_output=1, n_hidden=256):
        super(AttentionFirstAggregator, self).__init__()

        self.n_output = n_output
        self.al = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1), nn.Softmax(dim=1))
        self.pl = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))
        self.link = nn.Sigmoid()
        # self.link = nn.Identity()
    
    def forward(self,x):
        a = self.al(x)
        e = torch.sum(a*x, dim=1)
        o = self.pl(e)
        # o = self.link(o)
        return o

    def get_layers(self, x):
        a = self.al(x)
        p = torch.sigmoid(self.pl(x))
        e = torch.sum(a*x, dim=1)
        o = self.pl(e) 
        o = self.link(o)
        a = a.repeat(1,1,self.n_output) # one attention channel shared accross all classes
        return [o, a, p]

class AttentionLastAggregator(nn.Module):
    """
    """
    def __init__(self, n_features, n_output=1, n_hidden=256):
        super(AttentionLastAggregator, self).__init__()

        self.al = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output), nn.Softmax(dim=1))
        self.pl = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))
        # self.link = nn.Sigmoid()
        # self.link = nn.Identity()
    
    def forward(self,x):
        a = self.al(x)
        p = self.pl(x)
        o = torch.sum(a*p, dim=1)
        # o = self.link(o)
        return o

    def get_layers(self, x):
        a = self.al(x)
        p = self.pl(x)
        o = torch.sum(a*p, dim=1)
        # o = self.link(o)
        # p = self.link(p) # avail to produce probability instead of logits
        return [o, a, p] 

#
###
