import torch
import torch.nn as nn

class ImageReconstructor(nn.Module):
    def __init__(self):
        super(ImageReconstructor, self).__init__()
        
        # Encoder part: Convolutional layers to process patches of size 64x64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, 32, 32)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Reducing dimension further
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 8, 8)
        )

        # Use a dummy input to pass through the encoder to determine output size
        dummy_input = torch.zeros(1, 1, 64, 64)
        dummy_output = self.encoder(dummy_input)
        self.flattened_size = int(torch.numel(dummy_output)) * 4

        # Flatten the output of the convolutional layers
        self.flatten = nn.Flatten()

        # Fully connected layers to decode the features into the original image
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),  # Adjust the input size dynamically
            nn.ReLU(),
            nn.Linear(128, 128 * 128)  # Output vector to be reshaped to (1, 128, 128)
        )

    def forward(self, x):
        batch_size = x.size(0)
        patches = x.view(-1, 1, 64, 64)  # Reshape to treat each patch as a batch element

        encoded_patches = self.encoder(patches)
        flattened = self.flatten(encoded_patches)
        combined = flattened.view(batch_size, -1)  # Combine flattened encodings from all patches

        reconstructed = self.fc(combined)
        reconstructed_image = reconstructed.view(batch_size, 1, 128, 128)

        return reconstructed_image
