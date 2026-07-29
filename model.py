import torch
import torch.nn as nn


class GeometryExtractor(nn.Module):
    def __init__(self, latent_dim: int = 192, feature_out_dim: int = 1):
        super().__init__()
        self.t_vec = nn.Parameter(torch.zeros(latent_dim))
        self.rotation_matrix = nn.Parameter(torch.eye(latent_dim))
        self.lambda_vec = nn.Parameter(torch.ones(latent_dim))

        self.feature_extractor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_out_dim),
        )

    def forward(self, x):
        x = x + self.t_vec
        x = torch.matmul(x, self.rotation_matrix)
        x = x * self.lambda_vec
        return self.feature_extractor(x)