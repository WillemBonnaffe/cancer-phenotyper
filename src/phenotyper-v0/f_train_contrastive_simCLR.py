##############
## INITIATE ##
##############

## imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import random

#
###

###############
## FUNCTIONS ##
###############

class ProjectionHead(nn.Module):
    """ 
    """
    def __init__(self, embedding_dim, projection_dim):
        super(ProjectionHead, self).__init__() 
 
        ## Projection head (short)
        self.fc = nn.Linear(embedding_dim, projection_dim)  # Projects to the embedding dimension

    def forward(self, x):
        x = self.fc(x)
        return x

def simclr_loss(projections_a, projections_t, temperature=0.5):
    """ 
    """
    batch_size = projections_a.shape[0]
    
    ## Normalize projections to unit sphere
    projections_a = F.normalize(projections_a, dim=1)
    projections_t = F.normalize(projections_t, dim=1)
    
    ## Concatenate positive pairs
    projections = torch.cat([projections_a, projections_t], dim=0)  # (2*batch_size, feature_dim)
    
    ## Compute similarity matrix (cosine similarity)
    similarity_matrix = torch.mm(projections, projections.T)  # (2*batch_size, 2*batch_size)
    
    ## Create labels for contrastive learning
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = labels.to(projections.device)
    
    ## Mask to remove self-similarity (diagonal elements)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(projections.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
    
    ## Apply temperature scaling and compute cross-entropy loss
    logits = similarity_matrix / temperature
    loss = F.cross_entropy(logits, labels)
    
    return loss

def train(encoder, train_loader, test_loader, embedding_dim, projection_dim, num_epochs, device, temperature, learn_rate=0.01):
    """ 
    """

    ## Define the directory where you want to save your model checkpoints
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    ## Instantiate projection head
    projection_head = ProjectionHead(embedding_dim, projection_dim)
    projection_head.to(device)

    ## Optimizer: update both the encoder and the projection head
    optimizer = optim.SGD(list(encoder.parameters()) + list(projection_head.parameters()), lr=learn_rate)
    # optimizer = optim.SGD(list(encoder.parameters()) + list(projection_head.parameters()), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=0.001)

    ## Transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4, hue=0.4)
        ])
 
    ## Training loop
    train_loss_list = []
    test_loss_list = []
    progress_interval = 32
    for epoch in range(num_epochs):

        ## Training
        encoder.train()  # Set encoder to training mode
        projection_head.train()  # Set projection head to training mode
        total_train_loss = 0.0        
        for i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the parameter gradients
            if images.size(0) > 1:
    
                ## Forward pass to get embeddings
                images_a = images.to(device, dtype=torch.float)
                images_t = transform(images_a)
                embeddings_a = encoder(images_a)
                embeddings_t = encoder(images_t)
                projections_a = projection_head(embeddings_a)
                projections_t = projection_head(embeddings_t)
       
                ## Compute loss
                loss = simclr_loss(projections_a, projections_t, temperature)
                # loss = criterion(projections_a, projections_t) + penalisation(projections_a, projections_t) 
        
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
                if images.size(0) > 1:

                    # Forward pass to get embeddings
                    images_a = images.to(device, dtype=torch.float)
                    images_t = transform(images_a)
                    embeddings_a = encoder(images_a)
                    embeddings_t = encoder(images_t)
                    projections_a = projection_head(embeddings_a)
                    projections_t = projection_head(embeddings_t)
    
                    ## Compute loss
                    loss = simclr_loss(projections_a, projections_t, temperature)
                    # loss = criterion(projections_a, projections_t) + penalisation(projections_a, projections_t) 

                    ## Update total loss
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

        ## Assume `model` is your SimCLR model and `epoch` is the current epoch number
        checkpoint_path = f"{save_dir}/simclr_epoch_{epoch}.pth"
    
        ## Save every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.state_dict(),
            }, checkpoint_path)
            print(f"Model saved at epoch {epoch}: {checkpoint_path}")


    ## Return the fine-tuned encoder and both training and test loss lists
    return encoder, train_loss_list, test_loss_list

#
###
