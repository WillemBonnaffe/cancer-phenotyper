##### 
## ##
#####

##############
## INITIATE ##
##############

## Imports
import os
import numpy as np
import cv2
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

## Import modules
from .f_aggregator_clam import CLAM_SB as AggregatorArchitecture
from .f_dataset_embeds import InstanceEmbeddingsDataset
from .f_train import train_model
from .f_train import evaluate_model
from .f_train import grouped_stratified_kfold_split
from .f_train import get_auc_multilabel
from .f_utils import subset_list as sl
from .f_utils import format_label_file 

#
###

###############
## FUNCTIONS ##
###############

def main(pt_embeddings_file, pt_labels_file, pt_groups_file, pt_output_folder, selected_testing_fold, device, hp_list=[]):
    """
    """
    #########################
    ## USER DEFINED INPUTS ##

    ## Paths
    pt_folder_out = pt_output_folder
    pt_subfolder_out = pt_folder_out + f'hypertrain_fold_{selected_testing_fold}/'
    pt_models_folder = pt_subfolder_out + "models/"

    ## Parameters
    selected_testing_fold = selected_testing_fold - 1
    device = device

    ## Fixed parameters
    num_folds = 5
    num_hidden = 256
    num_repeats = 3

    ##############
    ## INITIATE ##

    ## Create output subfolder
    if os.path.exists(pt_output_folder) == False:
        os.mkdir(pt_output_folder)
 
    ## Create output subfolder
    if os.path.exists(pt_subfolder_out) == False:
        os.mkdir(pt_subfolder_out)
        
    ## Create models subfolder
    if os.path.exists(pt_models_folder) == False:
        os.mkdir(pt_models_folder)

    ## Get labels and groups and embeddings
    labels = format_label_file(pt_labels_file)
    groups = np.genfromtxt(pt_groups_file, dtype="str")
    embeddings = pickle.load(open(pt_embeddings_file, "rb"))

    ## Dependent parameters
    num_embeddings = embeddings[0].shape[1]

    ################
    ## HYPERTRAIN ##

    if hp_list != []:
        ## Initiate custom grid 
        batch_size_list = hp_list[0]
        num_epochs_list = hp_list[1] 
        learn_rate_list = hp_list[2]
    else:
        ## Initiate standard grid
        batch_size_list = [4, 8, 16, 32]
        num_epochs_list = [4, 8, 16, 32]
        learn_rate_list = [0.0001, 0.001]

    ## Collectors
    hypertrain_list = []

    ## Hypertrain loop
    for batch_size in batch_size_list:
        for num_epochs in num_epochs_list: 
            for learn_rate in learn_rate_list: 
               
                ## For each fold
                for selected_training_fold in range(num_folds):

                    ## For each repat
                    for repeat in range(num_repeats):

                        ## Model id
                        model_id = f"mod_agg_fold{selected_training_fold}_repeat{repeat}_bs{batch_size}_ne{num_epochs}_lr{learn_rate}.pth"

                        ## Check if checkpoint exists
                        if os.path.exists(pt_models_folder + model_id) == False:

                            ## Fit
                            model, train_loss, val_loss, test_loss = train_fold(
                                embeddings = embeddings, 
                                labels = labels, 
                                groups = groups, 
                                num_folds = num_folds, 
                                selected_training_fold = selected_training_fold, 
                                selected_testing_fold = selected_testing_fold, 
                                device = device, 
                                batch_size = batch_size, 
                                num_epochs = num_epochs, 
                                learn_rate = learn_rate,
                                num_embeddings = num_embeddings,
                                num_hidden = num_hidden
                                )

                            ## Save model
                            torch.save(model.state_dict(), pt_models_folder + model_id)

                        else:

                            ## Evaluate
                            model, train_loss, val_loss, test_loss = evaluate_fold(
                                pt_aggregator_in = pt_models_folder + model_id,
                                embeddings = embeddings, 
                                labels = labels, 
                                groups = groups, 
                                num_folds = num_folds, 
                                selected_training_fold = selected_training_fold, 
                                selected_testing_fold = selected_testing_fold, 
                                device = device, 
                                batch_size = batch_size, 
                                num_embeddings = num_embeddings,
                                num_hidden = num_hidden
                                )

                        ## Collect
                        hypertrain_list.append([selected_training_fold, repeat, batch_size, num_epochs, learn_rate, train_loss, val_loss, test_loss])
 
                        ## Update
                        print(f"Fold: {selected_training_fold + 1}, Repeat: {repeat + 1}, Hyperparameters: {batch_size:2d}, {num_epochs:2d}, {learn_rate:.4f}, Train auc: {np.mean(train_loss):.2f}, Validation auc: {np.mean(val_loss):.2f}, Test auc: {np.mean(test_loss):.2f}")

    ## Format and save
    pickle.dump(hypertrain_list, open(pt_subfolder_out + "obj_hypertrain_list.pkl", "wb"))

    ## Get best model
    get_tab_paths_best_models(pt_output_folder, selected_testing_fold + 1)

#
###

#######################
## SUPPORT FUNCTIONS ##
#######################

def train_fold(embeddings, labels, groups, num_folds, selected_training_fold, selected_testing_fold, device, batch_size, num_epochs, learn_rate, num_embeddings, num_hidden):
    """
    """
    ###########################
    ## PREPARE TRAINING DATA ##

    ## Determine test fold indices 
    trainval_folds, test_folds = grouped_stratified_kfold_split(labels, groups, num_folds)
    trainval_index = trainval_folds[selected_testing_fold-1]
    test_index = test_folds[selected_testing_fold-1]

    ## Checks
    # print(len(embeddings)) # DEBUG
    # print(selected_testing_fold)
    # print(trainval_index) # DEBUG
    # print(test_index) # DEBUG

    ## Determine validation fold indices
    train_folds, val_folds = grouped_stratified_kfold_split(labels[trainval_index], groups[trainval_index], num_folds)
    train_index = train_folds[selected_training_fold]
    val_index = val_folds[selected_training_fold]

    ## Split data in folds
    train_labels, val_labels, test_labels = sl(sl(labels, trainval_index), train_index), sl(sl(labels, trainval_index), val_index), sl(labels, test_index)
    train_embeddings, val_embeddings, test_embeddings = sl(sl(embeddings, trainval_index), train_index), sl(sl(embeddings, trainval_index), val_index), sl(embeddings, test_index)

    ## Create dataset
    num_classes = len(labels[0])
    train_dataset = InstanceEmbeddingsDataset(train_embeddings, train_labels, num_classes=num_classes)
    val_dataset = InstanceEmbeddingsDataset(val_embeddings, val_labels, num_classes=num_classes)
    test_dataset = InstanceEmbeddingsDataset(test_embeddings, test_labels, num_classes=num_classes)

    ## Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    ######################
    ## TRAIN AGGREGATOR ##

    ## Instantiate the model
    aggregator = AggregatorArchitecture(num_embeddings, num_classes, num_hidden)
    # if os.path.exists(pt_aggregator_in):
    #     aggregator.load_state_dict(torch.load(pt_aggregator_in))
    model = aggregator
   
    ## Define the loss function and optimizer
    pos_weight = torch.tensor((1-labels).sum(0)/labels.sum(0)).float().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(aggregator.parameters(), lr=learn_rate)
    ## DEBUG >>>print(f"positive weighting: {pos_weight.cpu().detach().numpy()}")

    ## Train the model
    model, train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, echo=False)

    #########################
    ## EVALUATE AGGREGATOR ##
    
    ## Evaluate
    outputs, _, _, true = evaluate_model(model, train_loader, device)
    auc_train = get_auc_multilabel(true, outputs)
    #
    outputs, _, _, true = evaluate_model(model, val_loader, device)
    auc_val = get_auc_multilabel(true, outputs)
    #
    outputs, _, _, true = evaluate_model(model, test_loader, device)
    auc_test = get_auc_multilabel(true, outputs)

    ## End 
    return model, auc_train, auc_val, auc_test 

def evaluate_fold(pt_aggregator_in, embeddings, labels, groups, num_folds, selected_training_fold, selected_testing_fold, device, batch_size, num_embeddings, num_hidden):
    """
    """
    ###########################
    ## PREPARE TRAINING DATA ##

    ## Determine test fold indices 
    trainval_folds, test_folds = grouped_stratified_kfold_split(labels, groups, num_folds)
    trainval_index = trainval_folds[selected_testing_fold-1]
    test_index = test_folds[selected_testing_fold-1]

    ## Determine validation fold indices
    train_folds, val_folds = grouped_stratified_kfold_split(labels[trainval_index], groups[trainval_index], num_folds)
    train_index = train_folds[selected_training_fold]
    val_index = val_folds[selected_training_fold]

    ## Split data in folds
    train_labels, val_labels, test_labels = sl(sl(labels, trainval_index), train_index), sl(sl(labels, trainval_index), val_index), sl(labels, test_index)
    train_embeddings, val_embeddings, test_embeddings = sl(sl(embeddings, trainval_index), train_index), sl(sl(embeddings, trainval_index), val_index), sl(embeddings, test_index)

    ## Create dataset
    num_classes = len(labels[0])
    train_dataset = InstanceEmbeddingsDataset(train_embeddings, train_labels, num_classes=num_classes)
    val_dataset = InstanceEmbeddingsDataset(val_embeddings, val_labels, num_classes=num_classes)
    test_dataset = InstanceEmbeddingsDataset(test_embeddings, test_labels, num_classes=num_classes)

    ## Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    #########################
    ## EVALUATE AGGREGATOR ##
    
    ## Instantiate the model
    aggregator = AggregatorArchitecture(num_embeddings, num_classes, num_hidden)
    if os.path.exists(pt_aggregator_in):
        aggregator.load_state_dict(torch.load(pt_aggregator_in))
    model = aggregator

    ## Evaluate
    outputs, _, _, true = evaluate_model(model, train_loader, device)
    auc_train = get_auc_multilabel(true, outputs)
    #
    outputs, _, _, true = evaluate_model(model, val_loader, device)
    auc_val = get_auc_multilabel(true, outputs)
    #
    outputs, _, _, true = evaluate_model(model, test_loader, device)
    auc_test = get_auc_multilabel(true, outputs)

    ## End 
    return model, auc_train, auc_val, auc_test 

def normalise(x):
    return (x - x.min())/(x.max() - x.min())

def collapse_columns_2Darray_to_string(arr, sep=""):
    """
    """
    arr_str = []
    for arr_ in arr:
        arr_ = [arr__ for arr__ in arr_.astype(str)]
        arr_ = sep.join(arr_)
        arr_str += [arr_]
    return np.array(arr_str)

def generate_model_id_from_hyperparameters(hp):
    """
    """
    return f"mod_agg_fold{int(hp[0])}_repeat{int(hp[1])}_bs{int(hp[2])}_ne{int(hp[3])}_lr{hp[4]}.pth"

def get_tab_paths_best_models(pt_input_folder, selected_testing_fold):
    """
    """

    #################
    ## FORMAT DATA ##
    
    ## Load data 
    pti = pt_input_folder + f"hypertrain_fold_{selected_testing_fold}/"
    ht = pickle.load(open(pti + "obj_hypertrain_list.pkl", "rb"))
    # headers: (selected_training_fold, repeat, batch_size, num_epochs, learn_rate, train_loss, val_loss, val_loss)
    
    ## Format to array
    hp = []
    train = []
    val = []
    test = []
    for ht_ in ht:
        hp += [ht_[0:5]]
        train += [ht_[5]]
        val += [ht_[6]]
        test += [ht_[7]]
    hp_1 = np.array(hp)
    train_1 = np.array(train)
    val_1 = np.array(val)
    test_1 = np.array(test)

    #############################
    ## PREPARE MODEL SELECTION ##
    
    ## Compute unique combinations of HPs
    hp_keys = collapse_columns_2Darray_to_string(hp_1[:,2:5], sep="_")
    hp_keys_unique = np.unique(hp_keys)
    hp_set_index = np.arange(len(hp_keys_unique))+1

    ###############################
    ## HYPERPARAMETERS SELECTION ##
    
    ## WIP >>
    # ## Target
    # target_arr = train_1
    # 
    # ## Assemble data
    # x_data = []
    # y_data = []
    # for hp_keys_unique_ in hp_keys_unique:
    #     s = np.argwhere(hp_keys == hp_keys_unique_)
    #     x_data += [hp_keys_unique_]
    #     y_data += [target_arr.mean(1)[s].flatten()]
    # y_data_train = y_data
    ## WIP <<

    ## Target
    target_arr = val_1
    
    ## Assemble data
    x_data = []
    y_data = []
    for hp_keys_unique_ in hp_keys_unique:
        s = np.argwhere(hp_keys == hp_keys_unique_)
        x_data += [hp_keys_unique_]
        y_data += [target_arr.mean(1)[s].flatten()]
    
    ## Compute mean AUC
    auc_list = [np.mean(y_) for y_ in y_data]
    
    ## Compute t-statistic
    t_list = [(np.mean(y_)-np.mean(y_data))/np.std(y_) for y_ in y_data]
    # t_list = [(np.mean(y_)-np.mean(y_data_train)) for y_ in y_data]
    
    ## Best HP set
    hp_star_index = np.argmax(t_list)
    # hp_star_index = np.argmin(t_list)
    hp_star = hp_keys_unique[hp_star_index]
    print(f"The best hyperparameter set is: {hp_star_index + 1}")
    print(f"HP star is: {hp_star}")
    
    ## Save
    hp_star_1 = hp_star

    ########################
    ## SELECT BEST MODELS ##
    
    ## Subset results for best models
    s_1 = np.argwhere(hp_star_1 == hp_keys)
    test_1_star = test_1[s_1].squeeze()
    hp_star_arr = np.array(hp_1[s_1][:,0])

    ##################
    ## FORMAT PATHS ##
    
    ## Get model ids
    model_id_list = [generate_model_id_from_hyperparameters(hp) for hp in hp_star_arr]
    
    ## Format path to models
    model_paths_list = [pti + 'models/' + model_id for model_id in model_id_list]
   
    ## Save
    np.savetxt(pti + 'tab_paths_best_models.txt', model_paths_list, fmt="%s")
       

#
###
