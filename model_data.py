import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import random

class ImagePatchDataset(Dataset):
    def __init__(self, image_paths, patchTransform=None,imageTransform=None):
        self.image_paths = image_paths
        self.imageTransform = imageTransform
        self.patchTransform = patchTransform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        image = self.imageTransform(image)
        patches = self.patchTransform(image)
        return image, patches

def create_patches(image, patch_size=64,num_patches=4):
    # Assuming image is 28x28 and patch_size is 14, we create 4 patches.
    patches = []
    max_x = image.size(1) - patch_size
    max_y = image.size(2) - patch_size

    for _ in range(num_patches):  # We need four patches
        # Randomly select the top-left pixel of the patch
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        # Extract the patch and add to the list of patches
        patch = image[:, x:x+patch_size, y:y+patch_size]
        patches.append(patch)

    # Optionally, you might want to concatenate these patches into a single tensor,
    # depending on how you want to feed them into your network. For instance:
    # patches = torch.cat(patches, dim=0)  # Concatenate patches along channel dimension
    # Return a stacked tensor of patches
    return torch.stack(patches, dim=0)

patchTransform = transforms.Compose([
    transforms.Lambda(lambda x: create_patches(x))
])

imageTransform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image to 128x128
    transforms.ToTensor(),
    # Add any additional transformations here (e.g., normalization)
    transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization
])