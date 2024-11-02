import torch
import torch.nn as nn
from ViT import VisionTransformer
from Legendre-Chebushev import KolmogorovArnoldNetwork

class AnomalyDetectionModel(nn.Module):
    def __init__(self, vit_params, kan_params):
        super(AnomalyDetectionModel, self).__init__()
        
        # Initialize Vision Transformer and KAN with given parameters
        self.vit = VisionTransformer(**vit_params)
        self.kan = KolmogorovArnoldNetwork(**kan_params)

    def forward(self, x):
        # Extract features using Vision Transformer
        features = self.vit(x)
        
        # Apply KAN to the extracted features
        anomaly_score = self.kan(features)
        
        return anomaly_score
