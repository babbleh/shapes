import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import wandb

from newmodel import ImageReconstructor
from model_data import ImagePatchDataset, create_patches, imageTransform, patchTransform

LEARNING_RATE = 0.003
EPOCHS = 500
DATASET = 'TESS_ISLANDS_1k'
BATCH_SIZE = 64
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "ENC_FL_DEC",
    "dataset": DATASET,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "loss_metric": "MSE",
    "optimizer": "Adam"
    }
)

def load_image_paths(base_dir):
    image_paths = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir) if fname.endswith('.png')]
    return image_paths

# Load all image paths
all_image_paths = load_image_paths('images/TessIslands1000')

# Split data into train and eval
train_paths, eval_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)

# Create Dataset instances
train_dataset = ImagePatchDataset(train_paths, imageTransform=imageTransform, patchTransform=patchTransform)
eval_dataset = ImagePatchDataset(eval_paths, imageTransform=imageTransform, patchTransform=patchTransform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Fetch one batch from the loader
device = torch.device('cpu')  # Explicitly using CPU
model = ImageReconstructor().to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, train_loader, epochs=EPOCHS):
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for labels, patches in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            reconstructed_images = model(patches)
            loss = criterion(reconstructed_images, labels)  # Compare reconstructed images to original images
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"loss": loss})
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

train(model, train_loader)
torch.save(model.state_dict(), 'model_state_dict.pth')
wandb.finish()
