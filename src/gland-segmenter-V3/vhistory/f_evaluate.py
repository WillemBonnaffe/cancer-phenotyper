###################
## f_evaluate.py ##
###################

## Goal: Functions to evaluate models.
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

## labels_to_onehot
## Goal: 
## Convert image with label values of segmentation classes to a one hot image with as many channels as classes.
## Inputs:
## img (numpy array): Image containing predicted segmentation classes.
## n_classes (int): Number of classes.
## Outputs:
## img_new (numpy array): One hot encoded image.
def labels_to_onehot(img, n_classes):
    img_new = np.zeros((img.shape[0], img.shape[1], n_classes))
    for i in range(n_classes):
        img_new[:, :, i] = ((img == i) * 1).squeeze()
    return img_new

## multiple_labels_to_onehot
## Goal: 
## Convert multiple images with label values of segmentation classes to one hot images with as many channels as classes.
## Inputs:
## images (numpy array): Array containing multiple images.
## n_classes (int): Number of semantic classes.
## drop_background (bool): Whether to remove background class (0).
## Outputs:
## labels_new (numpy array): Array of one hot encoded images.
def multiple_labels_to_onehot(images, n_classes, drop_background=False):
    labels_new = []
    for image in images:
        if drop_background==True:
            label_new_ = labels_to_onehot(image, n_classes)[:, :, 1:n_classes]
            labels_new.append(label_new_.reshape(1, label_new_.shape[0], label_new_.shape[1], label_new_.shape[2]))
        else:
            label_new_ = labels_to_onehot(image, n_classes)
            labels_new.append(label_new_.reshape(1, label_new_.shape[0], label_new_.shape[1], label_new_.shape[2]))
    labels_new = np.concatenate(labels_new, 0)
    return labels_new

## evaluate 
## Goal: 
## Evaluate the prediction of the model for each data point.
## Inputs:
## dataloader (torch object): Holds the features and targets.
## model (torch object): Model to evaluate.
## device (str): Device on which to perform the evaluation.
## Outputs:
## results (list): List containing actual, predicted, and input images.
def evaluate(dataloader, model, device):

    ## Initiate
    X = dataloader.dataset[:][0]
    Y = dataloader.dataset[:][1]
    input_image = []
    actual = []
    predictions = []

    ## For each input
    with torch.no_grad():
        for i in range(0, len(X)):

            ## Counter
            print(str(i) + "/" + str(len(X)))

            ## Load X 
            X_ = X[i]
            X_ = X_.reshape((1, X_.shape[0], X_.shape[1], X_.shape[2]))
            input_image_ = X_

            ## Load Y
            Y_ = Y[i]
            Y_ = Y_.reshape((1, 1, Y_.shape[0], Y_.shape[1])) # Only for cross entropy loss
            actual_ = Y_

            ## Get predictions
            predictions_ = model(X_.to(device)).cpu()

            ## Format predictions (logits -> probability -> labels)
            predictions_ = torch.argmax(torch.softmax(predictions_, dim=1), dim=1) # Only for cross entropy loss
            predictions_ = predictions_.reshape(predictions_.shape[0], 1, predictions_.shape[1], predictions_.shape[2]) # Only for cross entropy loss

            ## Convert to numpy
            input_image_ = input_image_.detach().numpy()
            actual_ = actual_.detach().numpy()
            predictions_ = predictions_.detach().numpy()

            ## Store
            input_image.append(input_image_)
            actual.append(actual_)
            predictions.append(predictions_)

        ## Format objects
        input_image = np.concatenate(input_image, 0)
        actual = np.concatenate(actual, 0)
        predictions = np.concatenate(predictions, 0)

        ## Convert to OpenCV images
        input_image = input_image.swapaxes(1, 2).swapaxes(2, 3)
        actual = actual.swapaxes(1, 2).swapaxes(2, 3)
        predictions = predictions.swapaxes(1, 2).swapaxes(2, 3)

        ## Convert to onehot encoding
        actual = multiple_labels_to_onehot(actual, n_classes=4, drop_background=True)
        predictions = multiple_labels_to_onehot(predictions, n_classes=4, drop_background=True)

        ## Check
        print(input_image.shape)
        print(actual.shape)
        print(predictions.shape)

    return [actual, predictions, input_image]

#
###
