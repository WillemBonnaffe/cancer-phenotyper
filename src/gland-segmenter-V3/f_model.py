################
## f_model.py ##
################

## Goal: Define a UNet model for semantic segmentation.
## Author: Willem Bonnaffe (w.bonnaffe@gmail.com) 

##############
## INITIATE ##
##############

import torch
import torch.nn as nn
import torch.nn.functional as F

#
###

#############
## CLASSES ##
#############

## DoubleConv
## Goal:
## Perform two consecutive convolutional operations with Batch Normalization and ReLU activation.
## Inputs:
## in_channels (int): Number of input channels.
## out_channels (int): Number of output channels.
## Outputs:
## x (torch.Tensor): Transformed tensor after double convolution.
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

## Down
## Goal:
## Perform downsampling using max pooling followed by a double convolution.
## Inputs:
## in_channels (int): Number of input channels.
## out_channels (int): Number of output channels.
## Outputs:
## x (torch.Tensor): Downsampled and transformed tensor.
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.pool_conv(x)
        return x

## Up
## Goal:
## Perform upsampling followed by a double convolution, with optional bilinear interpolation.
## Inputs:
## in_channels (int): Number of input channels.
## out_channels (int): Number of output channels.
## bilinear (bool): Whether to use bilinear interpolation for upsampling.
## Outputs:
## x (torch.Tensor): Upsampled and transformed tensor.
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        ## Upsampling layer
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        ## Double conv layer
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

## OutConv
## Goal:
## Perform a final 1x1 convolution to produce the desired number of output channels.
## Inputs:
## in_channels (int): Number of input channels.
## out_channels (int): Number of output channels.
## Outputs:
## x (torch.Tensor): Output tensor with the desired number of channels.    
class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

## UNet
## Goal:
## Define a U-Net model for semantic segmentation.
## Inputs:
## n_channels (int): Number of input channels.
## n_classes (int): Number of output classes.
## bilinear (bool): Whether to use bilinear upsampling.
## Outputs:
## logits (torch.Tensor): Output tensor with logits for each class.
class UNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 128)
        self.up1 = Up(256, 64)
        self.up2 = Up(128, 32)
        self.up3 = Up(64, 16)
        self.up4 = Up(32, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

#
###
