import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=2, dim=768, depth=12, heads=12, mlp_dim=3072):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim

        # Patch embedding layer
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, dim)

        # Transformer encoder layers with self-attention mechanism
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )

        # Multichannel autoencoder for spatial-temporal feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Divide the image into patches and flatten them
        x_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches = x_patches.contiguous().view(x_patches.size(0), -1, self.patch_size * self.patch_size * 3)

        # Apply patch embedding
        x_patches = self.patch_embedding(x_patches)

        # Add positional encoding (optional)
        x_patches += torch.arange(self.num_patches).unsqueeze(0).unsqueeze(2).to(x_patches.device)

        # Apply transformer encoder layers with self-attention mechanism
        x_patches = self.transformer_encoder(x_patches)

        # Multichannel autoencoder for spatial-temporal feature extraction
        encoded_features = self.encoder(x)
        decoded_features = self.decoder(encoded_features)

        # Combine features from transformer and autoencoder
        combined_features = x_patches.mean(dim=1) + encoded_features.view(encoded_features.size(0), -1)

        # Classification head
        output = self.classifier(combined_features)

        return output
