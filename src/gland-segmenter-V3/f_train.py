################
## f_train.py ##
################

## Goal: Functions to train models.
## Author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## Imports
import numpy as np
import torch

#
###

###############
## FUNCTIONS ##
###############

## dice_metric
## Goal: 
## Compute the DICE score.
## Source: 
## https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
## Inputs:
## inputs (torch.tensor): Model predictions.
## target (torch.tensor): Target values.
## Outputs:
## dice_score (float): DICE score.
def dice_metric(inputs, target): 
    intersection = 2.0 * (target * inputs).sum()
    union = (target * target).sum() + (inputs * inputs).sum()
    if (target * target).sum() == 0 and (inputs * inputs).sum() == 0:
        return 1.0
    return intersection / union

## dice_metric_channels
## Goal: 
## Compute the DICE score for each channel.
## Source: 
## https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
## Inputs:
## inputs (torch.tensor): Model predictions.
## target (torch.tensor): Target values.
## Outputs:
## dice_scores (torch.tensor): DICE scores for each channel.
def dice_metric_channels(inputs, target):
    intersection = 2.0 * (target * inputs).sum((1, 2))
    union = (target * target).sum((1, 2)) + (inputs * inputs).sum((1, 2))
    return intersection / union

## Goal: Compute DICE loss from logits.
## Source: https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
## Args:
## inputs (torch.tensor): Predictions of the model.
## target (torch.tensor): Target.
## Outputs:
## (torch.tensor): Dice loss from logits.    
def dice_with_logits_loss(inputs, target):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten()
    target = target.flatten()
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    return 1 - (intersection / union)

## train
## Goal: 
## Perform a train step.
## Inputs:
## dataloader (torch object): Holds the features and targets.
## model (torch object): Model to train.
## loss_fn (torch object): Loss function.
## optimizer (torch object): Optimizer function.
## device (str): Device on which to perform the training.
## Outputs:
## train_loss (float): Training loss value.
def train(dataloader, model, loss_fn, optimizer, device):

    ## Initiate
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    ## For each batch
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)        

        ## Compute prediction error
        pred = model(X)        
        loss = loss_fn(pred, Y)
             
        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
        ## Total batch loss (v1)
        with torch.no_grad():
            train_loss += loss_fn(pred, Y).item()

        ## Progress message every 10 batches
        if batch % 10 == 0:
            current = batch * len(X)
            print(f"Progress: [{current:>5}/{size:>5}]  Loss: {loss.item():>8f}")

    ## Update
    train_loss /= num_batches
    print(f"Train Error: {train_loss:>8f} \n")

    ## Terminate
    return train_loss

## test
## Goal: 
## Perform a test step.
## Inputs:
## dataloader (torch object): Holds the features and targets.
## model (torch object): Model to test.
## loss_fn (torch object): Loss function.
## device (str): Device on which to perform the testing.
## Outputs:
## test_loss (float): Testing loss value.
def test(dataloader, model, loss_fn, device):

    ## Initiate
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    ## For each batch
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()

    ## Update
    test_loss /= num_batches
    print(f"Test Error:  {test_loss:>8f} \n")

    ## Terminate
    return test_loss

#
###
