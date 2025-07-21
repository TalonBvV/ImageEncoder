import torch
import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=384):
        super().__init__()
        # Load a pretrained EfficientNet-B0
        self.base_model = timm.create_model(
            'mobilevit_xs',
            pretrained=True,
            num_classes=0  # Remove the final classification layer
        )
        
        # Get the number of output features from the base model
        num_features = self.base_model.num_features
        
        # Add a new head to project the features to our desired latent dimension
        self.latent_head = nn.Sequential(
            nn.Linear(num_features, latent_dim),
            nn.GELU(), # Activation function
            nn.LayerNorm(latent_dim) # Normalize the output
        )

    def forward(self, x):
        # x shape: (batch, 3, 128, 128)
        features = self.base_model(x) # Get features from EfficientNet
        # features shape: (batch, num_features)
        latent_vector = self.latent_head(features)
        # latent_vector shape: (batch, 384)
        return latent_vector
