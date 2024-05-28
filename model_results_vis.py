import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os 
from sklearn.model_selection import train_test_split

from newmodel import ImageReconstructor
from model_data import ImagePatchDataset, create_patches, imageTransform, patchTransform

def load_image_paths(base_dir):
    image_paths = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir) if fname.endswith('.png')]
    return image_paths

# Load all image paths
all_image_paths = load_image_paths('images/tess_islands')

# Split data into train and eval
train_paths, eval_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)

# Create Dataset instances
#train_dataset = ImagePatchDataset(train_paths, imageTransform=imageTransform, patchTransform=patchTransform)
eval_dataset = ImagePatchDataset(eval_paths, imageTransform=imageTransform, patchTransform=patchTransform)

#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

#load the model from the path model_state_dict.pth
model = ImageReconstructor()
model.load_state_dict(torch.load('model_state_dict.pth'))

device = torch.device('cpu')  # Explicitly using CPU
model = ImageReconstructor().to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss

#evaluate the model
def evaluate(model, eval_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for labels, patches in eval_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            reconstructed_images = model(patches)
            loss = criterion(reconstructed_images, labels)
            total_loss += loss.item()
    print(f'Evaluation Loss: {total_loss/len(eval_loader)}')

evaluate(model, eval_loader)
# Visualize the results
def visualize_results(loader, model, device):
    data_iter = iter(loader)
    original_images, patches = next(data_iter)

    original_images = original_images.to(device)
    patches = patches.to(device)

    reconstructed_images = model(patches).cpu().detach()

    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(15, 9))

    for i in range(5):
        axes[0, i].imshow(original_images[i][0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title("Original Image")
        axes[0, i].axis('off')

        # Assuming each original image has exactly 4 patches, concatenate them into a 2x2 grid
        # Each patch is of size (1, 64, 64)
        axes[1, i].imshow(reconstructed_images[i][0], cmap='gray')
        axes[1, i].set_title("Reconstructed Image")
        axes[1, i].axis('off')

        for j in range(4):
            axes[j+2, i].imshow(patches[i][j][0].cpu().numpy(), cmap='gray')
            axes[j+2, i].set_title(f"Patch {j+1}")
            axes[j+2, i].axis('off')

    plt.show()


visualize_results(eval_loader, model, device)