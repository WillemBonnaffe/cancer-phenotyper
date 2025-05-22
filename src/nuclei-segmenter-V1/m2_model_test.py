######################
## m2_model_test.py ##
######################

## goal: evaluate performance of model on train and validation set

## author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## import modules
from f_dataset import dataset
from f_model import Net
from f_UNet import UNet
# from f_model import UNet
from f_train import evaluate
from f_train import dice_metric

## imports
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import seaborn as sns
import pandas as pd
import torch

## arg parsers
parser = argparse.ArgumentParser()
parser.add_argument("--pt_dataset", default="/Volumes/SED/BDI/databases/segmentation/datasets/NucleiSeg/test/")
parser.add_argument("--pt_model", default="models/r13_2.pth")
parser.add_argument("--pt_output", default="out/r13_2.pth/")
args = parser.parse_args()

## load data
dataset_ = dataset(args.pt_dataset, tileSize=128, imgSize=128*8, thresholds=np.array([0, 100, 256]), n_repeat=1, n_skip_val=5, batchSize=1)
dataloader = dataset_.dataloader_t

## load model
device = "cpu"
print(f"Using {device} device")
model = UNet().to(device)
if os.path.exists(args.pt_model) == True:
    model.load_state_dict(torch.load(args.pt_model))
model.eval()

## paths
pto = args.pt_output
if os.path.exists(pto) == False:
    os.mkdir(pto)

#
###

###############
## FUNCTIONS ##
###############

## normalise
## goal: normalise array between 0 and 1
## array - np.array - to normalise
def normalise(array):
        return (array-np.min(array))/(np.max(array)-np.min(array))

#
###

##########
## MAIN ##
##########

##
## TRAINING ##

## predictions
output_images, predictions, input_images = evaluate(dataloader, model, device)

## visualise prediction
k = 0
for prediction in predictions:
    cv2.imwrite(pto + "train_" + str(k) + "_predicted.png", normalise(predictions[k])*255)
    cv2.imwrite(pto + "train_" + str(k) + "_output.png", normalise(output_images[k])*255)
    cv2.imwrite(pto + "train_" + str(k) + "_input.png", normalise(input_images[k])*255)
    k = k + 1

## compute DICE metric
dice_l = dice_metric(normalise(predictions), normalise(output_images))
print("")
print("-- Performance --")
print("DICE: " + str(np.round(dice_l,2)))
print("")

#
###
