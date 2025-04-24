###############################
## ENCODER CLASS DEFINITION  ##
###############################

## Goal: Define a Vision Transformer encoder for extracting embeddings from images.
## Author: Willem Bonnaffe (w.bonnaffe@gmail.com)

####################################
## IMPORTS AND LIBRARY DEPENDENCIES #
####################################

import torch
import torch.nn as nn
from torchvision import models

#
###

########################################
## VISION TRANSFORMER ENCODER CLASS  ##
########################################

## VisionTransformerEncoder
## Goal:
## Implement a basic Vision Transformer (ViT) encoder to extract feature embeddings from input images.
## Inputs:
## img_size (int): Size of the input images (assumed square).
## patch_size (int): Size of each patch.
## num_classes (int): Number of output classes.
## dim (int): Dimensionality of the embedding.
## depth (int): Number of transformer blocks.
## heads (int): Number of attention heads.
## mlp_dim (int): Dimensionality of the MLP in the transformer.
## dropout (float): Dropout rate.
## emb_dropout (float): Dropout rate for embeddings.
## Outputs:
## Embedding tensor for each input image.
class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=0, dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1):
        super(VisionTransformerEncoder, self).__init__()

        ## Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size  ## 3 for RGB channels
        self.patch_size = patch_size

        ## Linear projection of flattened patches
        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)

        ## Positional encoding
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        ## Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        ## Head for the output feature embeddings
        self.to_latent = nn.Identity()

    def forward(self, x):
        ## Extract patches from the input images
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(batch_size, -1, self.patch_dim)

        ## Apply the linear projection to the patches
        x = self.patch_to_embedding(x)

        ## Add class token and positional embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding[:, :(x.size(1))]
        x = self.dropout(x)

        ## Pass through the transformer layers
        x = self.transformer(x)

        ## Extract the class token as the embedding
        x = x[:, 0]
        return self.to_latent(x)

#
###

