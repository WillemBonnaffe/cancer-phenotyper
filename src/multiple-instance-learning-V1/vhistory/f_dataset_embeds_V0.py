#####
## ##
#####

## Goal:

##############
## INITIATE ##
##############

## Imports
import os
import numpy as np

#
###

###############################
## EMBEDDING DATASET CLASSES ##
###############################

class InstanceEmbeddingsDataset:
    """
    """
    def __init__(self, sample_list, labels, num_classes=1, padding=True):

        ## Properties
        self.sample_list = sample_list
        self.labels = labels
        self.num_classes = num_classes
        self.padding = padding

        ## Find the maximum number of instances in the samples
        self.max_num_instances = max(len(sample) for sample in sample_list)

        ## Find the number of embeddings in each instance
        self.num_embeddings = len(sample_list[0][0])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        """
        """
        sample = self.sample_list[index]
        labels = self.labels[index]

        ## Format embeddings 
        if self.padding:
            padded_sample = np.zeros((self.max_num_instances, self.num_embeddings))
            num_instances = len(sample)
            padded_sample[:num_instances, :] = sample
            sample = padded_sample

        ## Format labels
        # labels = np.array([labels], dtype=int)

        return sample, labels

#
###
