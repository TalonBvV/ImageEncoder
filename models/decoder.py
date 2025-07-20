import torch
import torch.nn as nn

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=384, output_channels=3, conditioning_dim=0):
        super().__init__()
        self.conditioning_dim = conditioning_dim
        
        # First, project the flat vector (plus conditioning) into a small spatial representation
        self.initial_projection = nn.Linear(latent_dim + conditioning_dim, 8 * 8 * 256)
        
        # Now, build the upsampling path
        self.upsampler = nn.Sequential(
            # Start from (batch, 256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 64x64
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1), # -> 128x128
            nn.Sigmoid() # Squash output pixel values to be between 0 and 1
        )

    def forward(self, x, conditioning=None):
        # x shape: (batch, 384)
        # conditioning shape: (batch, conditioning_dim)
        
        if self.conditioning_dim > 0:
            if conditioning is None:
                raise ValueError("Conditioning tensor is required for this decoder.")
            # Concatenate latent vector with conditioning information
            x = torch.cat((x, conditioning), dim=1)
            
        x = self.initial_projection(x)
        x = x.view(-1, 256, 8, 8) # Reshape into a spatial tensor
        reconstructed_image = self.upsampler(x)
        # reconstructed_image shape: (batch, 3, 128, 128)
        return reconstructed_image
