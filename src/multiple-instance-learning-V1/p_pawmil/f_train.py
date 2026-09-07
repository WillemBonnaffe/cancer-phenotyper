#####
## ##
#####

## Goal:

##############
## INITIATE ##
##############

## Imports
import time
import numpy as np
import torch
# from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

## Import modules
from .f_utils import collapse_columns_2Darray_to_string

#
###

####################
## FOLD FUNCTIONS ##
####################

def grouped_stratified_kfold_split(labels, groups, num_folds=3):
    """
    """
    ## Data
    X = np.ones(labels.shape[0]) # dummy data

    ## Format labels
    labels = collapse_columns_2Darray_to_string(labels)

    ## Initiate
    sgkf = StratifiedGroupKFold(n_splits=num_folds)

    ## Split 
    # sgkf.get_n_splits(X, labels)

    ## Get indices
    train_folds = []
    test_folds = []
    for i, (train_index, test_index) in enumerate(sgkf.split(X, labels, groups)):
        train_folds += [train_index]
        test_folds += [test_index]

    ## End
    return train_folds, test_folds

def custom_grouped_stratified_kfold_split(labels, groups, num_folds, send_to_test_th=3):
    """
    """
    ## Aggregate labels by column to produce unique labels
    unique_labels = np.unique(labels, axis=0)

    train_folds = []
    val_folds = []

    ## Iterate through each fold
    for fold_index in range(num_folds):

        train_indices = []
        val_indices = []
        send_to_train_counter = 0

        ## Iterate through unique labels and get corresponding indices
        for label in unique_labels:

            ## Get subset indices
            subset_indices = np.where((labels == label).all(axis=1))[0]
            subset_groups = groups[subset_indices]

            ## Get unique groups
            unique_groups = np.unique(subset_groups)

            # print(subset_indices)
            # print(subset_groups)
            # print(unique_groups)

            if len(unique_groups) >= num_folds:
                # print("enough groups")

                start_group_index = int(((fold_index / num_folds) * len(unique_groups)))
                end_group_index = int((((fold_index + 1) / num_folds) * len(unique_groups)))

                # print(start_group_index)
                # print(end_group_index)

                unique_groups_val_ = unique_groups[start_group_index:end_group_index]
                unique_groups_train_l_ = unique_groups[:start_group_index]
                unique_groups_train_r_ = unique_groups[end_group_index:]

                # print(unique_groups_val_)
                # print(unique_groups_train_l_)
                # print(unique_groups_train_r_)

                for unique_group_val_ in unique_groups_val_:
                    val_indices.extend(subset_indices[np.where(subset_groups == unique_group_val_)[0]])
                for unique_group_train_l_ in unique_groups_train_l_:
                    train_indices.extend(subset_indices[np.where(subset_groups == unique_group_train_l_)[0]])
                for unique_group_train_r_ in unique_groups_train_r_:
                    train_indices.extend(subset_indices[np.where(subset_groups == unique_group_train_r_)[0]])

            else:
                # print("less groups than folds")
                for unique_group in unique_groups:
                    if send_to_train_counter < send_to_test_th:
                        # print("send to train counter: " + str(send_to_train_counter))
                        # print(unique_group)
                        train_indices.extend(subset_indices[np.where(subset_groups == unique_group)[0]])
                        send_to_train_counter += 1
                    else:
                        # print("send to train counter: " + str(send_to_train_counter))
                        # print(unique_group)
                        val_indices.extend(subset_indices[np.where(subset_groups == unique_group)[0]])
                        send_to_train_counter = 0

        ## Merge train and validation subsets to form train and validation sets
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        ## Save fold
        train_folds.append(train_indices)
        val_folds.append(val_indices)

    return train_folds, val_folds

def stratified_kfold_split(labels, num_folds):
    """
    """
    ## Aggregate labels by column to produce unique labels
    unique_labels = np.unique(labels, axis=0)

    train_folds = []
    val_folds = []

    ## Iterate through each fold
    for fold_index in range(num_folds):

        train_indices = []
        val_indices = []

        ## Iterate through unique labels and get corresponding indices
        for label in unique_labels:
            subset_indices = np.where((labels == label).all(axis=1))[0]
            # np.random.shuffle(subset_indices)  # Shuffle indices for random selection

            start_index = int((fold_index / num_folds) * len(subset_indices))
            end_index = int(((fold_index + 1) / num_folds) * len(subset_indices))

            val_indices.extend(subset_indices[start_index:end_index])
            train_indices.extend(subset_indices[:start_index])
            train_indices.extend(subset_indices[end_index:])

        ## Merge train and validation subsets to form train and validation sets
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        ## Save fold
        train_folds.append(train_indices)
        val_folds.append(val_indices)

    return train_folds, val_folds

#
###

#####################
## TRAIN FUNCTIONS ##
#####################


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device="cpu", patience=90, echo=True):
    """
    """
    ## Initiate
    model = model.to(device)
    train_loss_trace = []
    val_loss_trace = []
    best_val_loss = float('inf')
    no_improvement_count = 0

    ## For each epoch
    for epoch in range(num_epochs):

        ## Train step
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        ## Test step
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)

        ## Update
        if echo == True:
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_loss_trace.append(train_loss)
        val_loss_trace.append(val_loss)

        ## Check if the validation loss has improved
        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"early stop")
                break

    ## End
    return model, train_loss_trace, val_loss_trace

def evaluate_model(model, dataloader, device):
    """
    """
    ## Initiate
    model = model.to(device)
    model.eval()
    output_1 = []
    output_2 = []
    output_3 = []
    true = []

    ## Evaluate
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            output_1_, output_2_, output_3_ = model.get_layers(inputs)
            output_1.append(output_1_.detach().cpu().numpy())
            output_2.append(output_2_.detach().cpu().numpy())
            output_3.append(output_3_.detach().cpu().numpy())
            true.append(labels.detach().cpu().numpy())

    ## Format and terminate
    output_1 = np.concatenate((output_1))
    output_2 = np.concatenate((output_2))
    output_3 = np.concatenate((output_3))
    true = np.concatenate((true))
    return output_1, output_2, output_3, true

def train_model_weighted(model, train_loader, val_loader, criterion, optimizer, num_epochs, device="cpu", patience=90, echo=True):
    """
    """
    ## Initiate
    model = model.to(device)
    train_loss_trace = []
    val_loss_trace = []
    best_val_loss = float('inf')
    no_improvement_count = 0

    ## For each epoch
    for epoch in range(num_epochs):

        ## Train step
        model.train()
        train_loss = 0.0

        for inputs, labels, weights in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            weights = weights.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = (loss * weights.to(device)).mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        ## Test step
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels, weights in val_loader:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = (loss * weights.to(device)).mean()
                val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)

        ## Update
        if echo == True:
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_loss_trace.append(train_loss)
        val_loss_trace.append(val_loss)

        ## Check if the validation loss has improved
        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"early stop")
                break

    ## End
    return model, train_loss_trace, val_loss_trace

def evaluate_model_weighted(model, dataloader, device):
    """
    """
    ## Initiate
    model = model.to(device)
    model.eval()
    output_1 = []
    output_2 = []
    output_3 = []
    true = []

    ## Evaluate
    with torch.no_grad():
        for inputs, labels, weights in dataloader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            output_1_, output_2_, output_3_ = model.get_layers(inputs)
            output_1.append(output_1_.detach().cpu().numpy())
            output_2.append(output_2_.detach().cpu().numpy())
            output_3.append(output_3_.detach().cpu().numpy())
            true.append(labels.detach().cpu().numpy())

    ## Format and terminate
    output_1 = np.concatenate((output_1))
    output_2 = np.concatenate((output_2))
    output_3 = np.concatenate((output_3))
    true = np.concatenate((true))
    return output_1, output_2, output_3, true

#
###

#####################
## QUANTIFICATIONS ##
#####################

def get_auc_multilabel(y_true, y_scores):
    """
    """
    roc_auc = []
    num_labels = y_scores.shape[1]
    for label_idx in range(num_labels):
        fpr, tpr, _ = roc_curve(y_true[:, label_idx], y_scores[:, label_idx])
        roc_auc += [auc(fpr, tpr)]
    return roc_auc

def get_auc_multilabel_grouped(y_true, y_scores, groups):
    """
    Calculate group-level AUC for multilabel predictions.
    Predictions and labels are averaged within each group.
    """
    groups = np.array(groups)

    # Get unique groups
    unique_groups = np.unique(groups)

    # Aggregate samples to group level
    y_true_grouped = np.array([
        y_true[groups == group].mean(axis=0)
        for group in unique_groups
    ])

    y_scores_grouped = np.array([
        y_scores[groups == group].mean(axis=0)
        for group in unique_groups
    ])

    # Calculate AUC for each label
    roc_auc = []
    num_labels = y_scores.shape[1]

    for label_idx in range(num_labels):
        fpr, tpr, _ = roc_curve(
            y_true_grouped[:, label_idx],
            y_scores_grouped[:, label_idx]
        )
        roc_auc.append(auc(fpr, tpr))

    return roc_auc
    
#
###

#############################
## VISUALISATION FUNCTIONS ##
#############################

def plot_loss(train_loss, val_loss, pt_save):
    """
    """
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(pt_save)
    plt.close()

def plot_roc(y_true, y_scores, pt_save):
    """
    """
    plt.figure()
    num_labels = y_scores.shape[1]
    for label_idx in range(num_labels):
        fpr, tpr, _ = roc_curve(y_true[:, label_idx], y_scores[:, label_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(pt_save)
    plt.close()

def plot_roc_single_label(y_true, y_scores, pt_save):
    """
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(pt_save)
    plt.close()

def plot_predictions(y_true, y_pred, pt_save):
    """
    """
    num_labels = y_true.shape[1]

    plt.figure()
    for label_idx in range(num_labels):
        plt.scatter(y_true[:, label_idx], y_pred[:, label_idx], label=f'Label {label_idx}')
        plt.plot([np.min(y_true[:, label_idx]), np.max(y_true[:, label_idx])],
                 [np.min(y_true[:, label_idx]), np.max(y_true[:, label_idx])], 'k--', lw=2)

    plt.xlabel('True Label')
    plt.ylabel('Model Prediction')
    plt.title('Model Predictions vs True Labels')
    plt.legend()
    plt.savefig(pt_save)
    plt.close()

def plot_predictions_single_label(y_true, y_pred, pt_save):
    """
    """
    plt.scatter(y_true, y_pred)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)
    plt.xlabel('True Label')
    plt.ylabel('Model Prediction')
    plt.title('Model Predictions vs True Labels')
    plt.savefig(pt_save)
    plt.close()

#
###
