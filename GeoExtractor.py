import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.utils.parametrizations as parametrizations

"""
Here is the current version of GeometryExtractor and GeoExtractorTrainer for further experiments. Note that during the project we used older versions 
and training loops in a 'makeshift' notebooks. These code can be found in the notebooks folder. From now on, we will use this version for strick reproducibility.
"""


class GeometryExtractor(nn.Module):
    def __init__(self, latent_dim,feature_out_dim, implicit_orthogonal=False):
        super(GeometryExtractor, self).__init__()

        self.t_vec = nn.Parameter(torch.zeros(latent_dim))
        self.rotation_matrix = nn.Parameter(torch.eye(latent_dim))
        
        # New conditionally applied implicit orthogonal parameterization
        if implicit_orthogonal:
            parametrizations.orthogonal(self, "rotation_matrix")
            
        self.lambda_vec = nn.Parameter(torch.ones(latent_dim))

        # 4. Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_out_dim)
        )

    def forward(self, x):
        x = x + self.t_vec
        x = torch.matmul(x, self.rotation_matrix)
        x = x * self.lambda_vec

        # Feature Extraction
        logits = self.feature_extractor(x)
        return logits


class GeoExTrainer:
    def __init__(self, target_lambda=2, verbose=2, n_epochs=100, beta_coeff=1.35, alpha_coeff=1.35, out_dim=1, implicit_orthogonal=False):
        self.target_lambda = target_lambda
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.beta_coeff = beta_coeff
        self.alpha_coeff = alpha_coeff
        self.out_dim = out_dim
        self.implicit_orthogonal = implicit_orthogonal

    def train(self, X, y):
        dataset = TensorDataset(X, y.unsqueeze(1) if y.ndim == 1 else y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256)

        # --- 3. Training Setup ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate the model with the implicit_orthogonal flag
        H_model = GeometryExtractor(
            latent_dim=192,  
            feature_out_dim=self.out_dim,
            implicit_orthogonal=self.implicit_orthogonal
        ).to(device)
        
        optimizer = optim.Adam(H_model.parameters(), lr=1e-3)

        # Hyperparameters for Regularization
        alpha_l1 = 0.01  # Sparsity strength
        beta_ortho = 0.01 # Orthogonality strength

        # --- 4. Training Loop ---
        epochs = self.n_epochs
        # Threshold for considering a lambda coefficient "active"
        sparsity_threshold = 1e-2

        if self.verbose > 0:
            print(f"{'Epoch':<6} | {'Total':<8} | {'MAE':<8} | {'L1':<8} | {'Ortho':<8} | {'Non-Zero'}")
            print("-" * 80)

        for epoch in range(epochs):
            H_model.train()
            # Metrics trackers
            running_mae, running_l1, running_ortho, running_total = 0, 0, 0, 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # Forward pass
                preds = H_model(batch_X)

                # Loss Components
                mae_loss = F.l1_loss(preds, batch_y) # For monitoring

                sorted_vec, _ = torch.sort(H_model.lambda_vec, descending=True)
                l1_reg = torch.norm(sorted_vec[self.target_lambda:], p=1)

                R = H_model.rotation_matrix
                eye = torch.eye(R.size(0), device=device)
                ortho_dist = torch.norm(torch.matmul(R, R.t()) - eye, p=2)

                # Optimization Goal
                total_loss = mae_loss + (alpha_l1 * l1_reg) + (beta_ortho * ortho_dist)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Accumulate for printing
                running_total += total_loss.item()
                running_mae += mae_loss.item()
                running_l1 += alpha_l1 * l1_reg.item()
                running_ortho += beta_ortho * ortho_dist.item()

            # Calculate Sparsity (count non-zero lambda coefficients)
            with torch.no_grad():
                non_zero_count = (H_model.lambda_vec.abs() > sparsity_threshold).sum().item()
                diff = torch.norm(torch.matmul(R, R.t()) - eye, p='fro')
                det = torch.det(R)

            if epoch == 1:
                n = len(train_loader)
                perc_mae = 100 * running_mae / running_total
                perc_l1 = 100 * running_l1 / running_total
                
                # Added tiny epsilon to prevent division by zero in case implicit parametrization makes ortho_dist strictly 0
                perc_ortho = max(100 * running_ortho / running_total, 1e-8) 
                
                alpha_l1 = alpha_l1 * 5 / perc_l1
                beta_ortho = beta_ortho * 5 / perc_ortho

            if epoch % 5 == 0:
                n = len(train_loader)
                perc_mae = 100 * running_mae / running_total
                perc_l1 = 100 * running_l1 / running_total
                perc_ortho = 100 * running_ortho / running_total
                if non_zero_count > self.target_lambda:
                    alpha_l1 = self.alpha_coeff * alpha_l1
                if diff > 1 or torch.abs(det-1) > 0.05:
                    beta_ortho = self.beta_coeff * beta_ortho
                if self.verbose > 1:
                    print(f"{epoch:<6} | {running_total/n:<8.4f}  "
                          f"{perc_mae:<8.4f} | {perc_l1:<8.4f} | {perc_ortho:<8.4f} | "
                          f"{non_zero_count}/{H_model.lambda_vec.size(0)}")
                          
        if self.verbose > 0:
            n = len(train_loader)
            R = H_model.rotation_matrix
            eye = torch.eye(R.size(0), device=device)
            perc_mae = 100 * running_mae / running_total
            perc_l1 = 100 * running_l1 / running_total
            perc_ortho = 100 * running_ortho / running_total
            Orthonorm = torch.norm(torch.matmul(R, R.t()) - eye, p=2).item()
            Det = torch.det(R).item()
            print(f"{epoch:<6} | {running_total/n:<8.4f}  "
                  f"{perc_mae:<8.4f} | {perc_l1:<8.4f} | {perc_ortho:<8.4f} | "
                  f"{non_zero_count}/{H_model.lambda_vec.size(0)}")
            print('Dimensionality: ' + str(non_zero_count))
            print('Difference from orthogonality: ' + str(Orthonorm))
            print('Determinant of the rotation: ' + str(Det))
            
        mae_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = H_model(batch_X)
                mae_loss += F.l1_loss(preds, batch_y)
            mae_loss /= len(test_loader)
            
        if self.verbose > 0:
            print(f"Test MAE: {mae_loss.item()}")
            
        metrics = [mae_loss, non_zero_count, Det, Orthonorm]
        return H_model, metrics
