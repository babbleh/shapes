import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random

class ImageReconstructor(nn.Module):
    def __init__(self):
        super(ImageReconstructor, self).__init__()
        
        # Encoder part: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, 7, 7)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 7, 7)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, 3, 3)
        )

        self.additional_layers = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Further reducing dimension
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Further reducing dimension
        )

        self.flatten = nn.Flatten()

        # Recalculate the flattened size based on the output of additional_layers
        self.flattened_size = 64 * 12 * 12  # Assuming further dimension reduction
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),  # Adjusted the input size
            nn.ReLU()
        )


        # Combine and process encoded patches
        self.fc = nn.Sequential(
            nn.Linear(64*32*3*3, 128),  # Combine all encoded patches
            nn.ReLU(),
            nn.Linear(128, 28 * 28)  # Output vector to be reshaped to (1, 28, 28)
        )

        # Decoder part: Reconstructs the full image from the encoded state
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # Output: (1, 14, 14)
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        patches = x.view(-1, 1, 10, 10)  # Reshape to treat each patch as a batch element

        encoded_patches = self.encoder(patches)
        flattened = self.flatten(encoded_patches)
        combined = flattened.view(batch_size, -1)  # Combine flattened encodings from all patches

        reconstructed = self.fc(combined)
        reconstructed_image = reconstructed.view(batch_size, 1, 28, 28)

        # Optionally, you might decode to improve image quality
        # reconstructed_image = self.decoder(reconstructed_image.view(batch_size, 32, 7, 7))

        return reconstructed_image
