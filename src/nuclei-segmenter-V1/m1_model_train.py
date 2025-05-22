#######################
## m1_model_train.py ##
#######################

## goal: main script to train model

## author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## import modules
from f_dataset import dataset
from f_model import Net
from f_UNet import UNet
# from f_model import UNet
from f_train import train
from f_train import test
from f_train import dice_loss
from f_train import dice_with_logits_loss

## imports
import numpy as np
import os
import torch
from torch import nn
import argparse

## arg parsers
parser = argparse.ArgumentParser()
parser.add_argument("--pt_dataset", default="/Volumes/SED/BDI/databases/segmentation/datasets/NucleiSeg/train/")
parser.add_argument("--pt_model_in", default="models/r13_2.pth")
parser.add_argument("--pt_model_out", default="models/r13_2.pth")
parser.add_argument("--n_epochs", default=10, type=int)
args = parser.parse_args()

## load data
dataset_ = dataset(args.pt_dataset, tileSize=128, imgSize=128*8, thresholds=np.array([0, 100, 256]), n_repeat=16, n_skip_val=5, batchSize=1)
dataloader_l = dataset_.dataloader_l
dataloader_t = dataset_.dataloader_t

## load model
device = "mps"
print(f"Using {device} device")
model = UNet().to(device)
if os.path.exists(args.pt_model_in) == True:
    model.load_state_dict(torch.load(args.pt_model_in))

## loss function
# loss_fn   = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.BCELoss()
# loss_fn = dice_loss
loss_fn = dice_with_logits_loss

## optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#
###

##############
## TRAINING ##
##############

## training
print("training starts...")
for t in range(args.n_epochs):
    print(f"-- Epoch {t+1} ------------- \n")
    train_loss = train(dataloader_l, model, loss_fn, optimizer, device)
    test(dataloader_t, model, loss_fn, device)
    if train_loss <= 0.1:
        print("early termination")
        break

print("training complete")
torch.save(model.state_dict(), args.pt_model_out)

#
###
