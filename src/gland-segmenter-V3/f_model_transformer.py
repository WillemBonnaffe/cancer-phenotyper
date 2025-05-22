############################
## f_model_transformer.py ##
############################

## Goal: Define a UNet-Transfomer model for semantic segmentation.
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

## TransformerEncoderLayer
## Goal:
## Define a single encoder layer for a Transformer model with self-attention and feed-forward network.
## Inputs:
## embed_dim (int): Dimension of the embedding.
## num_heads (int): Number of attention heads.
## mlp_dim (int): Dimension of the feed-forward network.
## dropout (float): Dropout rate.
## Outputs:
## x (torch.Tensor): Transformed tensor after attention and MLP operations.
class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attention, _ = self.attention(x, x, x)
        x = x + attention
        x = self.norm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x

## SegmentTransformer
## Goal:
## Implement a Transformer model for image segmentation, including positional encoding and Transformer layers.
## Inputs:
## input_size (int): Size of the input image.
## embed_dim (int): Dimension of the embedding.
## depth (int): Number of Transformer layers.
## num_heads (int): Number of attention heads.
## mlp_dim (int): Dimension of the feed-forward network.
## dropout (float): Dropout rate.
## Outputs:
## x (torch.Tensor): Transformed tensor after the Transformer layers.
class SegmentTransformer(nn.Module):
    
    def __init__(self, input_size=64, embed_dim=128, depth=1, num_heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.num_patches = input_size * input_size
        # self.pos_embed = torch.zeros(1, embed_dim, self.num_patches) + torch.arange(0, self.num_patches)[None, None, :]
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.num_patches)) # For learnable positional encoding
        self.pos_drop = nn.Dropout(dropout)
        self.transformer = nn.Sequential(*[TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)])

    def forward(self, x): # (B, C, H, W)
        B, C, H, W = x.size()
        x = x.reshape(B, C, H * W) # (B, C, H, W) --> (B, C, P)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = torch.permute(x, (2, 0, 1))  # (B, C, P) --> (P, B, C) as this is the convention for multihead attention
        x = self.transformer(x)
        x = torch.permute(x, (1, 2, 0))  # (P, B, C) --> (B, C, P)
        x = x.reshape(B, C, H, W) # (B, C, P) --> (B, C, H, W) 
        return x

## UNetTransformer
## Goal:
## Combine a U-Net architecture with a Transformer encoder for enhanced semantic segmentation performance.
## Inputs:
## n_channels (int): Number of input channels.
## n_classes (int): Number of output classes.
## bilinear (bool): Whether to use bilinear upsampling.
## input_size (int): Size of the input images.
## embed_dim (int): Dimension of the embedding.
## depth (int): Number of Transformer layers.
## num_heads (int): Number of attention heads.
## mlp_dim (int): Dimension of the feed-forward network.
## dropout (float): Dropout rate.
## Outputs:
## logits (torch.Tensor): Output tensor with logits for each class.
class UNetTransformer(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, input_size=64, embed_dim=128, depth=2, num_heads=8, mlp_dim=256, dropout=0.1):
        super(UNetTransformer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32) # H/2
        self.down2 = Down(32, 64) # H/4
        self.down3 = Down(64, 128) # H/8
        self.down4 = Down(128, 128) # H/16 (e.g. 1024 -> 512 -> 256 -> 128 -> 64)
        self.transf = SegmentTransformer(input_size, embed_dim, depth, num_heads, mlp_dim, dropout)
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
        x5 = self.transf(x5) # Transformer
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

#
###
