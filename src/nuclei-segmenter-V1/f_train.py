################
## f_train.py ##
################

## goal: functions to train and evaluate models 

## author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## imports
import numpy as np
import torch

## import modules
from f_dataset import labels_to_onehot
from f_dataset import multiple_labels_to_onehot

#
###

###############
## FUNCTIONS ##
###############

## normalise
## goal: cast values of array between 0 and 1
## array - np.array - to normalise
def normalise(array):
    return (array-np.min(array))/(np.max(array)-np.min(array))

## dice loss
## goal: compute DICE loss
## source: https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
## inputs - torch.tensor - predictions of model
## target - torch.tensor - target
def dice_loss(inputs, target):
    inputs = inputs.flatten()
    target = target.flatten()
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    return 1 - (intersection / union)

## dice loss
## goal: compute DICE loss
## source: https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
## inputs - torch.tensor - predictions of model
## target - torch.tensor - target
def dice_with_logits_loss(inputs, target):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten()
    target = target.flatten()
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    return 1 - (intersection / union)

## train function
## goal: function to perform a train step
## dataloader - torch object - holds the features and targets 
## model - torch object - model
## loss_fn - torch object - loss function
## optimizer - torch object - optimizer function
## device - string - device on which to perform the training
def train(dataloader, model, loss_fn, optimizer, device):

    ## initiate
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    ## for each batch
    for batch, (Y, X) in enumerate(dataloader):
        Y, X = Y.to(device), X.to(device)

        ## compute prediction error
        pred = model(X)
        loss = loss_fn(pred, Y)
             
        ## backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
        ## total batch loss(v1)
        with torch.no_grad():
            train_loss += loss_fn(pred, Y).item()

    ## update
    train_loss /= num_batches
    print(f"Train Error: {train_loss:>8f} \n")

    ## terminate
    return train_loss

## test function
## goal: function to perform a test step
## dataloader - torch object - holds the features and targets 
## model - torch object - model
## loss_fn - torch object - loss function
## optimizer - torch object - optimizer function
## device - string - device on which to perform the training
def test(dataloader, model, loss_fn, device):

    ## initiate
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    ## for each batch
    with torch.no_grad():
        for Y, X in dataloader:
            Y, X = Y.to(device), X.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()

    ## update
    test_loss /= num_batches
    print(f"Test Error:  {test_loss:>8f} \n")

    ## terminate
    return test_loss

## evaluate 
## goal: function to evaluate the prediction of the model for each data point
## dataloader - torch object - holds the features and targets 
## model - torch object - model
## device - string - device on which to perform the training
def evaluate(dataloader, model, device):

    ## initiate
    Y = dataloader.dataset[:][0]
    X = dataloader.dataset[:][1]
    input_image = []
    actual = []
    predictions = []

    ## for each input
    with torch.no_grad():
        for i in range(0,len(X)):

            ## counter
            print(str(i) + "/" + str(len(X)))

            ## load X 
            X_ = X[i]
            X_ = X_.reshape((1, X_.shape[0], X_.shape[1], X_.shape[2]))
            input_image_ = X_

            ## load Y
            Y_ = Y[i]
            Y_ = Y_.reshape((1, Y_.shape[0], Y_.shape[1], Y_.shape[2]))
            # Y_ = Y_.reshape((1, 1, Y_.shape[0], Y_.shape[1])) # only for cross entropy loss
            actual_ = Y_

            ## get predictions
            predictions_ = model(X_.to(device)).cpu()

            ## format predictions (logits -> probability -> labels)
            predictions_ = (torch.sigmoid(predictions_) > 0.99)*1 # only for BCE from logits
            # predictions_ = torch.argmax(torch.softmax(predictions_, dim=1), dim=1) # only for cross entropy loss
            # predictions_ = predictions_.reshape(predictions_.shape[0], 1, predictions_.shape[1], predictions_.shape[2]) # only for cross entropy loss

            ## convert to numpy
            input_image_ = input_image_.detach().numpy()
            actual_ = actual_.detach().numpy()
            predictions_ = predictions_.detach().numpy()

            ## store
            input_image.append(input_image_)
            actual.append(actual_)
            predictions.append(predictions_)

        ## format objects
        input_image = np.concatenate(input_image, 0)
        actual = np.concatenate(actual, 0)
        predictions = np.concatenate(predictions, 0)

        ## convert to opencv images
        input_image = input_image.swapaxes(1,2).swapaxes(2,3)
        actual = actual.swapaxes(1,2).swapaxes(2,3)
        predictions = predictions.swapaxes(1,2).swapaxes(2,3)

        # ## convert to onehot encoding
        # actual = multiple_labels_to_onehot(actual, n_classes=4, drop_background=True)
        # predictions = multiple_labels_to_onehot(predictions, n_classes=4, drop_background=True)

        ## check
        print(input_image.shape)
        print(actual.shape)
        print(predictions.shape)

    return [actual, predictions, input_image]

## WIP >>
# def model_predict(X, model, device):
# 
#     ## remove gradients
#     with torch.no_grad():
# 
#         ## format X 
#         X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
#         X = X.swapaxes(3,2).swapaxes(2,1)
#         X = torch.Tensor(X)
# 
#         ## get predictions
#         X = model(X.to(device)).cpu()
# 
#         ## format predictions (logits -> probability -> labels)
#         X = torch.argmax(torch.softmax(X, dim=1), dim=1) 
#         X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
# 
#         ## convert to numpy
#         X = X.detach().numpy()
# 
#         ## convert to opencv images
#         X = X.swapaxes(1,2).swapaxes(2,3)
# 
#         ## convert to onehot encoding
#         X = labels_to_onehot(X[0], n_classes=4)[:,:,1:4] * 255
# 
#     return X
## WIP <<

## dice_metric
## goal: compute the DICE score
## source: https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
## inputs - torch.tensor - model predictions
## target - torch.tensor - target
def dice_metric(inputs, target): 
    inputs = inputs.flatten()
    target = target.flatten()
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return intersection / union

#
###

