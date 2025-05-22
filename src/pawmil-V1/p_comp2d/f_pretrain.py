##############
## INITIATE ##
##############

## imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import random

#
###

###############
## FUNCTIONS ##
###############

def train(encoder, train_loader, test_loader, embedding_dim, projection_dim, num_epochs, device, margin, alpha=0.001):
   
    ## Criterion for contrastive learning
    criterion = FastTripletDistanceLoss(margin)

    ## Regularization to penalize self-correlations
    penalisation = SelfCorrelationLoss(alpha)

    ## Instantiate projection head
    projection_head = ProjectionHead(embedding_dim, projection_dim)
    projection_head.to(device)

    ## Optimizer: update both the encoder and the projection head
    optimizer = optim.SGD(list(encoder.parameters()) + list(projection_head.parameters()), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=0.001)

    ## Transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
        ])
 
    ## Training loop
    train_loss_list = []
    test_loss_list = []
    progress_interval = 100
    for epoch in range(num_epochs):

        ## Training
        encoder.train()  # Set encoder to training mode
        projection_head.train()  # Set projection head to training mode
        total_train_loss = 0.0
        for i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the parameter gradients
    
            ## Forward pass to get embeddings
            images_a = images.to(device, dtype=torch.float)
            images_t = transform(images_a)
            embeddings_a = encoder(images_a)
            embeddings_t = encoder(images_t)
            projections_a = projection_head(embeddings_a)
            projections_t = projection_head(embeddings_t)
   
            ## Compute loss
            loss = criterion(projections_a, projections_t) + penalisation(projections_a.t())
    
            ## Backpropagation and optimization
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimizer step
    
            total_train_loss += loss.item()

            ## Progress message
            if (i + 1) % progress_interval == 0:
                avg_loss = total_train_loss / (i + 1)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], "
                      f"Avg Loss: {avg_loss:.4f}")

        ## Logging training loss
        train_loss_list.append(total_train_loss / len(train_loader))

        ## Testing
        encoder.eval()  # Set encoder to evaluation mode
        projection_head.eval()  # Set projection head to evaluation mode
        total_test_loss = 0.0
        with torch.no_grad():  # Disable gradient tracking during inference
            for i, (images, _) in enumerate(test_loader):

                # Forward pass to get embeddings
                images_a = images.to(device, dtype=torch.float)
                images_t = transform(images_a)
                embeddings_a = encoder(images_a)
                embeddings_t = encoder(images_t)
                projections_a = projection_head(embeddings_a)
                projections_t = projection_head(embeddings_t)

                ## Compute loss
                loss = criterion(projections_a, projections_t) + penalisation(projections_a.t())

                total_test_loss += loss.item()

                ## Progress message
                if (i + 1) % progress_interval == 0:
                    avg_loss = total_test_loss / (i + 1)
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(test_loader)}], "
                          f"Avg Loss: {avg_loss:.4f}")

        ## Logging test loss
        test_loss_list.append(total_test_loss / len(test_loader))

        ## Print progress
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss_list[-1]:.4f}, Test Loss: {test_loss_list[-1]:.4f}")

    ## Return the fine-tuned encoder and both training and test loss lists
    return encoder, train_loss_list, test_loss_list

#
###

#############
## CLASSES ##
#############

class ProjectionHead(nn.Module):

    def __init__(self, embedding_dim, projection_dim):
        super(ProjectionHead, self).__init__() 
 
        ## projection head (short)
        self.fc = nn.Linear(embedding_dim, projection_dim)  # Projects to the embedding dimension

    def forward(self, x):
        x = self.fc(x)
        return x

class FastTripletDistanceLoss(nn.Module):

    def __init__(self, margin):
        super(FastTripletDistanceLoss, self).__init__()
        self.margin = margin 
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, anchor, transformed):

        ## pairwise distances 
        pairwise_distances_p = self.pdist(anchor, transformed)
        pairwise_distances_n = self.pdist(anchor, torch.flip(transformed,[0]))

        ## normalised 
        normalised_distances_p = pairwise_distances_p/pairwise_distances_p.max()
        normalised_distances_n = pairwise_distances_n/pairwise_distances_n.max()

        # Compute the loss
        loss = torch.relu(normalised_distances_p.mean() - normalised_distances_n.mean() + self.margin)
        return loss

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, embeddings):
        # Compute similarity scores (dot product between normalized embeddings)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        scores = torch.matmul(embeddings, embeddings.t()) # / temperature
        loss = scores.mean()
        return loss

class CorrelationLoss(nn.Module):
    def __init__(self, alpha):
        super(CorrelationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, embeddings_a, embeddings_t):
        # Compute similarity scores (dot product between normalized embeddings)
        embeddings_a = (embeddings_a - embeddings_a.mean(0))/embeddings_a.std(0)
        embeddings_t = (embeddings_t - embeddings_t.mean(0))/embeddings_t.std(0)
        scores = torch.matmul(embeddings_a, embeddings_t.t())**2 
        loss = scores.mean() * self.alpha 
        return loss

class SelfCorrelationLoss(nn.Module):
    def __init__(self, alpha):
        super(SelfCorrelationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, embeddings):
        # Compute similarity scores (dot product between normalized embeddings)
        embeddings = (embeddings - embeddings.mean(0))/embeddings.std(0)
        # embeddings = ((embeddings.t() - embeddings.mean(1))/embeddings.std(1)).t()
        scores = torch.matmul(embeddings, embeddings.t())**2 
        loss = scores.mean() * self.alpha 
        return loss

#
###
