################
## f_model.py ##
################

## goal: functions and classes to define model architecture

## author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## imports
import torch
from torch import nn

#
###

#############
## CLASSES ##
#############

## double convolution module
class DoubleConv(nn.Module):
    def __init__(self, n_i, n_o):
        super().__init__()
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels=n_i, out_channels=n_i, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=n_i,track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=n_i, out_channels=n_o, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=n_o,track_running_stats=False),
                nn.ReLU()
                )
    def forward(self, x):
        return self.double_conv(x)
 
## model
class Net(nn.Module):

    def __init__(self, I=3, W=1, O=1):
        super(Net,self).__init__()
        self.cv_input      = DoubleConv(    I,  4*W)
        self.cv_down_1     = DoubleConv(  4*W,  8*W) 
        self.cv_down_2     = DoubleConv(  8*W, 16*W) 
        self.cv_down_3     = DoubleConv( 16*W, 32*W) 
        self.cv_mid        = DoubleConv( 32*W, 32*W) 
        self.cv_up_3       = DoubleConv( 32*W, 16*W) 
        self.cv_up_2       = DoubleConv( 16*W,  8*W) 
        self.cv_up_1       = DoubleConv(  8*W,  4*W) 
        self.cv_output     = DoubleConv(  4*W,    O)
        self.fu_downsample = nn.MaxPool2d(2,stride=2)
        self.fu_upsample   = nn.Upsample(scale_factor=2)
        self.fu_link       = nn.Softmax(dim=1)
    
    def forward(self,x0):
        x1  = self.cv_input(x0)
        e1  = self.cv_down_1(x1)
        e1  = self.fu_downsample(e1)
        e2  = self.cv_down_2(e1)
        e2  = self.fu_downsample(e2)
        e3  = self.cv_down_3(e2)
        e3  = self.fu_downsample(e3)
        x2  = self.cv_mid(e3)
        d3  = self.cv_up_3(x2)
        d3  = self.fu_upsample(d3)
        d2  = self.cv_up_2(d3)
        d2  = self.fu_upsample(d2)
        d1  = self.cv_up_1(d2)
        d1  = self.fu_upsample(d1)
        x3  = self.cv_output(d1)
        return x3

    def getNetFeatures(self,x0):
        return [x0, x1,  e1, e2, e3, x2, d3, d2, d1, x3] 


## model
class UNet(nn.Module):

    def __init__(self, I=3, W=4, O=1):
        super(UNet,self).__init__()
        self.W = W
        self.cv_input      = DoubleConv(    I,  4*W)
        self.cv_down_1     = DoubleConv(  4*W,  8*W)
        self.cv_down_2     = DoubleConv(  8*W, 16*W)
        self.cv_down_3     = DoubleConv( 16*W, 32*W)
        self.cv_mid        = DoubleConv( 32*W, 32*W)
        self.cv_up_3       = DoubleConv( 32*W,  8*W)
        self.cv_up_2       = DoubleConv( 16*W,  4*W)
        self.cv_up_1       = DoubleConv(  8*W,  2*W)
        self.cv_output     = DoubleConv(  4*W,    O)
        self.fu_downsample = nn.MaxPool2d(2,stride=2)
        self.fu_upsample   = nn.Upsample(scale_factor=2)

    def forward(self,x0):
        x1  = self.cv_input(x0)
        e1  = self.cv_down_1(x1)
        e1_  = self.fu_downsample(e1)
        e2  = self.cv_down_2(e1_)
        e2_  = self.fu_downsample(e2)
        e3  = self.cv_down_3(e2_)
        e3_  = self.fu_downsample(e3)
        x2  = self.cv_mid(e3_)
        d3  = self.cv_up_3(x2)
        d3  = self.fu_upsample(d3)
        d3  = torch.cat((d3,e3[:,0:(8*self.W)]),dim=1)
        d2  = self.cv_up_2(d3)
        d2  = self.fu_upsample(d2)
        d2  = torch.cat((d2,e2[:,0:(4*self.W)]),dim=1)
        d1  = self.cv_up_1(d2)
        d1  = self.fu_upsample(d1)
        d1  = torch.cat((d1,e1[:,0:(2*self.W)]),dim=1)
        x3  = self.cv_output(d1)
        return x3

    def getNetFeatures(self,x0):
        return [x0, x1,  e1, e2, e3, x2, d3, d2, d1, x3]

#
###
