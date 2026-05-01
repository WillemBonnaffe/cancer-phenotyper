##### 
## ##
#####

##############
## INITIATE ##
##############

## Imports
import os
import cv2
import time
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

## Import modules
from .f_dataset_embeds import InstanceEmbeddingsDataset
from .f_train import evaluate_model
from .f_aggregator_pawmil import AttentionLastAggregator as AggregatorArchitecture
from .f_utils import format_path_file
from .f_utils import format_label_file 

#
###

#######################
## SUPPORT FUNCTIONS ##
#######################

def compute_ensemble_mean_sd(scores_ensemble):
    """
    """
    scores_ensemble_mu = []
    scores_ensemble_sd = []
    for scores in scores_ensemble:
        scores_ensemble_mu += [scores.mean(2)]
        scores_ensemble_sd += [scores.std(2)]
    return scores_ensemble_mu, scores_ensemble_sd

def compute_ensemble_quantiles(scores_ensemble, q=[0.05,0.5,0.95]):
    """
    """
    scores_ensemble_qlow = []
    scores_ensemble_qmid = []
    scores_ensemble_qhig = []
    for scores in scores_ensemble:
        scores_ensemble_qlow += [np.quantile(scores,q[0],2)]
        scores_ensemble_qmid += [np.quantile(scores,q[1],2)]
        scores_ensemble_qhig += [np.quantile(scores,q[2],2)]
    return scores_ensemble_qlow, scores_ensemble_qmid, scores_ensemble_qhig

def load_and_format_ensemble(pti):
    """
    """
    ensemble = []
    model_folders = os.listdir(pti)
    for model_folder in model_folders:
        ensemble += [pickle.load(open(pti + model_folder + "/obj_paw_scores.pkl", "rb"))]

    ## Parameters
    num_models = len(ensemble)
    num_samples = len(ensemble[0])
    num_instances_, num_labels = ensemble[0][0].shape
    # dims: (num_models, num_samples, num_instances, num_labels)

    ## Re-arrange to have number of models last
    ensemble_new = []
    for j in range(num_samples):
        ensemble_new += [np.array([ensemble[i][j] for i in range(num_models)]).transpose(1,2,0)]
    ensemble = ensemble_new
    # dims: (num_samples, num_instances, num_labels, num_models)

    ## Compute mean and standard deviation
    ensemble_mu, ensemble_sd = compute_ensemble_mean_sd(ensemble)
    ensemble_low, ensemble_mid, ensemble_hig = compute_ensemble_quantiles(ensemble)

    return ensemble_mu, ensemble_low, ensemble_hig

def format_label_file(label_file):
    """
    """
    ## read labels
    labels = np.genfromtxt(label_file)

    ## format labels
    if len(labels.shape) == 1:
        labels = labels.reshape(-1,1)

    return labels


def load_format_all(pti):
    """
    """

    ## Load labels
    labels = format_label_file(pti + "tab_labels.txt")
    GT_labels = labels

    ## Load WSI paths
    paths = format_path_file(pti + "tab_paths.txt")

    ## Load WSI paths
    groups = format_path_file(pti + "tab_groups.txt")

    ## Load coordinates
    coords = pickle.load(open(pti + "encoder/obj_coordinates.pkl", "rb"))

    ## Load embeddings
    embeds = pickle.load(open(pti + "encoder/obj_embeddings.pkl", "rb"))

    ## Load ensemble
    ensemble_mid, ensemble_low, ensemble_hig = load_and_format_ensemble(pti + "aggregator/")

    ## Initiate
    scores_mu_ = []
    scores_low_ = []
    scores_hig_ = []
    embeds_ = []
    coords_ = []
    paths_ = []
    labels_ = []
    groups_ = []

    ## For each sample
    for i in range(len(paths)):

        ## Variables
        num_instances = ensemble_mid[i].shape[0]
        num_labels = ensemble_mid[i].shape[1]

        ## Collect instance-level objects
        scores_mu_ += [ensemble_mid[i]]
        scores_low_ += [ensemble_low[i]]
        scores_hig_ += [ensemble_hig[i]]
        embeds_ += [embeds[i]]
        coords_ += [coords[i]]

        ## Collect sample-level objects
        paths_ += [np.array(paths[i]).repeat(num_instances).reshape(-1,1)]
        labels_ += [np.array(labels[i]).reshape(-1,num_labels).repeat(num_instances,0)]
        groups_ += [np.array(groups[i]).repeat(num_instances).reshape(-1,1)]

    ## Format data
    embeds = np.concatenate(embeds_)
    scores_mu = np.concatenate(scores_mu_)
    scores_low = np.concatenate(scores_low_)
    scores_hig = np.concatenate(scores_hig_)
    coords = np.concatenate(coords_)
    paths = np.concatenate(paths_)
    labels = np.concatenate(labels_)
    groups = np.concatenate(groups_)

    pickle.dump(embeds, open(pti + "obj_embeds.pkl", "wb"))
    pickle.dump(scores_mu, open(pti + "obj_scores_mu.pkl", "wb"))
    pickle.dump(scores_low, open(pti + "obj_scores_low.pkl", "wb"))
    pickle.dump(scores_hig, open(pti + "obj_scores_hig.pkl", "wb"))
    pickle.dump(coords, open(pti + "obj_coords.pkl", "wb"))
    pickle.dump(paths, open(pti + "obj_paths.pkl", "wb"))
    pickle.dump(labels, open(pti + "obj_labels.pkl", "wb"))
    pickle.dump(groups, open(pti + "obj_groups.pkl", "wb"))

#
###

####################
## MAIN FUNCTIONS ##
####################

def main_apply_aggregator_ensemble(pt_input_folder, selected_testing_fold, rpt_model_paths_file, device):
    """
    """
    ## Get model paths
    pt_models_list = format_path_file(pt_input_folder + rpt_model_paths_file)

    ## For each model in path list
    for pt_model in pt_models_list:
        main_apply_aggregator_single(pt_input_folder, selected_testing_fold, pt_model, device)

    ## Load and format all data
    # load_format_all(pt_input_folder)

def main_apply_aggregator_single(pt_embeddings_file, pt_labels_file, pt_model_file, pt_output_folder, device):
# def main_apply_aggregator_single(pt_input_folder, selected_testing_fold, pt_model, device):
    """
    """
    #############################
    ## USER DEFINED PARAMETERS ##

    ## Paths
    pt_embeddings_in = pt_embeddings_file
    pt_aggregator_in = pt_model_file 
    pt_label_file = pt_labels_file 

    ## Parameters
    device = device

    ## Fixed parameters
    num_hidden = 256
    batch_size = 1

    ##############
    ## INITIATE ##

    ## Create output subfolder
    if os.path.exists(pt_output_folder) == False:
        os.mkdir(pt_output_folder)

    ## Get labels
    labels = format_label_file(pt_label_file)
    num_classes = len(labels[0])

    ## Load embeddings
    embeddings = pickle.load(open(pt_embeddings_in, "rb"))

    ## Dependent parameters
    num_embeddings = embeddings[0].shape[1]

    ################
    ## AGGREGATOR ##

    ## Instantiate the model
    aggregator = AggregatorArchitecture(num_embeddings, num_classes, num_hidden)
    if os.path.exists(pt_aggregator_in):
        aggregator.load_state_dict(torch.load(pt_aggregator_in))
        print("loaded model")
        
    #########################
    ## EVALUATE AGGREGATOR ##

    ## For each sample
    t0 = time.time()
    outputs = []
    attention_scores = []
    prediction_scores = []
    paw_scores = []
    ground_truth = []
    for i in range(len(embeddings)):

        ## Prepare data for evaluation
        all_dataset = InstanceEmbeddingsDataset([embeddings[i]], [labels[i]], padding=False, num_classes=num_classes, subsample=False)
        all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

        ## Evaluate
        outputs_, attention_scores_, prediction_scores_, ground_truth_ = evaluate_model(aggregator, all_loader, device)

        ## Format (remove batch dimension)
        attention_scores_ = attention_scores_[0]
        prediction_scores_ = prediction_scores_[0]

        ## Compute paw-scores
        paw_scores_ = attention_scores_ * prediction_scores_
        # paw_scores_ = 1/(1+np.exp(-(attention_scores_ * prediction_scores_)))

        ## Store
        outputs.append(outputs_)
        attention_scores.append(attention_scores_)
        prediction_scores.append(prediction_scores_)
        paw_scores.append(paw_scores_)
        ground_truth.append(ground_truth_)

    tf = time.time()
    print(f"generated scores in {tf-t0:.2f}s")

    ## Format attention and prediction and save
    pickle.dump(attention_scores, open(pt_output_folder + "obj_attention_scores.pkl","wb"))
    pickle.dump(prediction_scores, open(pt_output_folder + "obj_prediction_scores.pkl", "wb"))
    pickle.dump(paw_scores, open(pt_output_folder + "obj_paw_scores.pkl", "wb"))

#
###
