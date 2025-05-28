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
    Dataset class for handling instances of embeddings with optional padding and subsampling.
    """
    def __init__(self, sample_list, labels, num_classes=1, padding=True, subsample=True, subsample_size=1000):
        """
        Parameters:
        - sample_list: List of samples, where each sample contains multiple instances (each a list/array of embeddings).
        - labels: Corresponding labels for each sample.
        - num_classes: Number of classes for the labels (default 1).
        - padding: Boolean to enable/disable padding (default True).
        - subsample: Boolean to enable/disable random subsampling of instances during initialization (default False).
        - subsample_size: If subsampling is enabled, the number of instances to randomly select from each sample.
        """

        # Apply subsampling during initialization if enabled
        if subsample and subsample_size is not None:
            self.sample_list = [
                self._subsample_instances(sample, subsample_size)
                for sample in sample_list
            ]
        else:
            self.sample_list = sample_list

        self.labels = labels
        self.num_classes = num_classes
        self.padding = padding

        # Find the maximum number of instances in the samples for padding
        self.max_num_instances = max(len(sample) for sample in self.sample_list)

        # Find the number of embeddings per instance
        self.num_embeddings = len(self.sample_list[0][0])

    def _subsample_instances(self, sample, subsample_size):
        """
        Helper method to randomly subsample instances from a sample.
        """
        num_instances = len(sample)
        if subsample_size < num_instances:
            sample_indices = np.random.choice(num_instances, subsample_size, replace=False)
            return [sample[i] for i in sample_indices]
        else:
            return sample

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        """
        Get a sample and its label. Optionally pad the sample.
        """
        sample = self.sample_list[index]
        label = self.labels[index]

        # Padding the sample to max_num_instances if enabled
        if self.padding:
            padded_sample = np.zeros((self.max_num_instances, self.num_embeddings))
            num_instances = len(sample)
            padded_sample[:num_instances, :] = sample
            sample = padded_sample

        return sample, label

#
###
