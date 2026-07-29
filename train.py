import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import GeometryExtractor


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def geometry_extraction(
    X: torch.Tensor,
    y: torch.Tensor,
    target_lambda: int = 2,
    verbose: int = 2,
    n_epochs: int = 150,
    beta_coeff: float = 1.35,
    alpha_coeff: float = 1.35,
    out_dim: int = 1,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[GeometryExtractor, List[float]]:
    """Train GeometryExtractor with L1 + orthogonality regularization."""
    if seed is not None:
        set_seed(seed)

    if device is None:
        device = "cpu"
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TensorDataset(X, y.unsqueeze(1) if y.ndim == 1 else y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = GeometryExtractor(latent_dim=X.shape[1], feature_out_dim=out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    alpha_l1 = 0.01
    beta_ortho = 0.01
    sparsity_threshold = 1e-2

    if verbose > 0:
        print(f"{'Epoch':<6} | {'Total':<8} | {'MAE':<8} | {'L1':<8} | {'Ortho':<8} | {'Non-Zero'}")
        print("-" * 80)

    for epoch in range(n_epochs):
        model.train()
        running_mae = running_l1 = running_ortho = running_total = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            preds = model(batch_X)
            mae_loss = F.l1_loss(preds, batch_y)

            sorted_vec, _ = torch.sort(model.lambda_vec, descending=True)
            l1_reg = torch.norm(sorted_vec[target_lambda:], p=1)

            R = model.rotation_matrix
            eye = torch.eye(R.size(0), device=device)
            ortho_dist = torch.norm(torch.matmul(R, R.t()) - eye, p=2)

            total_loss = mae_loss + (alpha_l1 * l1_reg) + (beta_ortho * ortho_dist)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_total += total_loss.item()
            running_mae += mae_loss.item()
            running_l1 += alpha_l1 * l1_reg.item()
            running_ortho += beta_ortho * ortho_dist.item()

        with torch.no_grad():
            non_zero_count = int((model.lambda_vec.abs() > sparsity_threshold).sum().item())
            diff = torch.norm(torch.matmul(R, R.t()) - eye, p="fro")
            det = float(torch.det(R).item())

        # adaptive regularization scaling (epoch 1)
        if epoch == 1:
            n = len(train_loader)
            perc_l1 = 100 * running_l1 / running_total
            perc_ortho = 100 * running_ortho / running_total
            if perc_l1 > 0:
                alpha_l1 = alpha_l1 * 5 / perc_l1
            if perc_ortho > 0:
                beta_ortho = beta_ortho * 5 / perc_ortho

        # adaptive regularization scaling (every 5 epochs)
        if epoch % 5 == 0:
            n = len(train_loader)
            perc_mae = 100 * running_mae / running_total
            perc_l1 = 100 * running_l1 / running_total
            perc_ortho = 100 * running_ortho / running_total

            if non_zero_count > target_lambda:
                alpha_l1 = alpha_coeff * alpha_l1
            if diff > 1 or abs(det - 1) > 0.05:
                beta_ortho = beta_coeff * beta_ortho

            if verbose > 1:
                print(
                    f"{epoch:<6} | {running_total / n:<8.4f}  "
                    f"{perc_mae:<8.4f} | {perc_l1:<8.4f} | {perc_ortho:<8.4f} | "
                    f"{non_zero_count}/{model.lambda_vec.size(0)}"
                )

    # final stats
    if verbose > 0:
        n = len(train_loader)
        R = model.rotation_matrix
        eye = torch.eye(R.size(0), device=device)
        perc_mae = 100 * running_mae / running_total
        perc_l1 = 100 * running_l1 / running_total
        perc_ortho = 100 * running_ortho / running_total
        orthonorm = float(torch.norm(torch.matmul(R, R.t()) - eye, p=2).item())
        print(
            f"{epoch:<6} | {running_total / n:<8.4f}  "
            f"{perc_mae:<8.4f} | {perc_l1:<8.4f} | {perc_ortho:<8.4f} | "
            f"{non_zero_count}/{model.lambda_vec.size(0)}"
        )
        print(f"Dimensionality: {non_zero_count}")
        print(f"Difference from orthogonality: {orthonorm}")
        print(f"Determinant of the rotation: {det}")

    # test MAE
    test_mae = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            test_mae += F.l1_loss(model(batch_X), batch_y).item()
        test_mae /= len(test_loader)
    if verbose > 0:
        print(f"Test MAE: {test_mae}")

    metrics = [test_mae, non_zero_count, det, orthonorm]
    return model, metrics


def probe_training(
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 50,
    verbose: int = 2,
    out_dim: int = 1,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[GeometryExtractor, List[float]]:
    """Simple probe training without geometric regularization."""
    if seed is not None:
        set_seed(seed)

    if device is None:
        device = "cpu"
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TensorDataset(X, y.unsqueeze(1) if y.ndim == 1 else y)
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = GeometryExtractor(latent_dim=X.shape[1], feature_out_dim=out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(n_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            loss = F.l1_loss(model(batch_X), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_mae = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            test_mae += F.l1_loss(model(batch_X), batch_y).item()
        test_mae /= len(test_loader)

    if verbose > 0:
        print(f"Test MAE: {test_mae}")

    return model, [test_mae]