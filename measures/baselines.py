import numpy as np
import torch

def random_rotation(dim: int, device: str = "cpu") -> torch.Tensor:
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return torch.tensor(Q, device=device, dtype=torch.float32)

def get_baseline(dim: int, device: str = "cpu") -> dict:
    return {
        "R": random_rotation(dim, device),
        "t": torch.zeros(dim, device=device),
        "l": torch.zeros(dim, device=device),
    }
