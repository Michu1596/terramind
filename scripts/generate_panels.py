#!/usr/bin/env python3
import os
import glob
import ast
import argparse
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize

# --- Circular Statistics Helpers ---

def circular_mean(angles):
    return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))

def circular_correlation(alpha, beta):
    mu_alpha = circular_mean(alpha)
    mu_beta = circular_mean(beta)
    numerator = np.sum(np.sin(alpha - mu_alpha) * np.sin(beta - mu_beta))
    denominator = np.sqrt(
        np.sum(np.sin(alpha - mu_alpha) ** 2) * np.sum(np.sin(beta - mu_beta) ** 2)
    )
    return numerator / denominator if denominator != 0 else 0.0

def mean_resultant_length(phi, theta):
    delta = theta - phi
    C = np.mean(np.cos(delta))
    S = np.mean(np.sin(delta))
    return np.sqrt(C**2 + S**2)

def neg_abs_circ_corr(center, points, labels):
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    if np.any((np.abs(dx) < 1e-12) & (np.abs(dy) < 1e-12)):
        return 0.0
    phi = np.arctan2(dy, dx)
    corr = circular_correlation(phi, labels)
    if np.isnan(corr):
        return 0.0
    return -abs(corr)

def find_optimal_center(points, labels, init_guesses=None):
    # Bound the optimization strictly within/near the manifold bounding box
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    dx = max(x_max - x_min, 1e-5)
    dy = max(y_max - y_min, 1e-5)
    
    bounds = [
        (x_min - 0.5 * dx, x_max + 0.5 * dx),
        (y_min - 0.5 * dy, y_max + 0.5 * dy)
    ]

    if init_guesses is None:
        mean_c = np.mean(points, axis=0)
        bbox_c = (points.min(axis=0) + points.max(axis=0)) / 2.0
        init_guesses = [mean_c, bbox_c]

    best_center = None
    best_corr = -1.0
    
    for guess in init_guesses:
        guess_clipped = np.clip(guess, [b[0] for b in bounds], [b[1] for b in bounds])
        result = minimize(
            neg_abs_circ_corr,
            guess_clipped,
            args=(points, labels),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )
        corr = -result.fun
        if corr > best_corr:
            best_corr = corr
            best_center = result.x

    phi_opt = np.arctan2(points[:, 1] - best_center[1], points[:, 0] - best_center[0])
    best_mrl = mean_resultant_length(phi_opt, labels)
    return best_center, best_corr, best_mrl


# --- Feature Configurations ---

VARIABLES = {
    "h":    {"code": "H",    "title": "Altitude",   "meta_idx":  3, "circular": False, "time": False},
    "l":    {"code": "L",    "title": "Latitude",   "meta_idx":  0, "circular": False, "time": False},
    "long": {"code": "Long", "title": "Longitude",  "meta_idx":  1, "circular": True,  "time": False},
    "t":    {"code": "T",    "title": "Time",       "meta_idx": -1, "circular": True,  "time": True },
}


def find_transformation_file(transformations_dir, seed, var_code):
    candidate_dirs = [
        transformations_dir,
        os.path.join(".", transformations_dir),
        os.path.join("..", transformations_dir),
        "transformations",
        "../transformations",
    ]
    
    search_dirs = []
    for d in candidate_dirs:
        if os.path.exists(d) and d not in search_dirs:
            search_dirs.append(d)

    for d in search_dirs:
        patterns = [
            os.path.join(d, f"{var_code}_geometry_result_seed{seed}.csv"),
            os.path.join(d, f"{var_code}_geometry_results_seed{seed}.csv"),
            os.path.join(d, f"{var_code}*seed{seed}*.csv"),
            os.path.join(d, f"*{var_code}*seed{seed}*.csv"),
            os.path.join(d, f"*{seed}*.csv"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
                
    return None


def resolve_data_file(data_dir, filename):
    candidate_paths = [
        os.path.join(data_dir, filename),
        os.path.join(".", "data", filename),
        os.path.join("..", "data", filename),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate 2D Affine Subspace feature panels across network depths.")
    parser.add_argument("--seed", type=str, required=True, help="Seed identifier")
    parser.add_argument("--feature", type=str, required=True, choices=["t", "h", "long", "l"], help="Feature name (t, h, long, l)")
    parser.add_argument("--iterations", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10, 12], help="Iterations/depths to include")
    parser.add_argument("--transformations_dir", type=str, default="transformations", help="Path to transformations directory")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="results_panels", help="Path to output directory")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device (cpu or cuda)")
    
    args = parser.parse_args()
    
    feature_key = args.feature.lower()
    var_config = VARIABLES[feature_key]
    var_code = var_config["code"]
    var_title = var_config["title"]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    trans_file = find_transformation_file(args.transformations_dir, args.seed, var_code)
    if not trans_file:
        raise FileNotFoundError(
            f"Could not find transformation CSV for seed '{args.seed}' and feature '{var_code}' in {args.transformations_dir}"
        )
    print(f"📁 Using transformation file: {trans_file}")
        
    df_geom = pd.read_csv(trans_file)
    for col in ["rotation_matrix", "translation_vec", "lambda_vec"]:
        df_geom[col] = df_geom[col].apply(ast.literal_eval)

    plot_data = []
    all_labels = []

    for iter_val in args.iterations:
        latents_file = f"latents_{iter_val}_92160_for_intervention.pt"
        latents_path = resolve_data_file(args.data_dir, latents_file)
        
        if not latents_path:
            print(f"⚠️  Latents file missing for iteration {iter_val}: {latents_file}. Skipping.")
            continue
            
        data = torch.load(latents_path, map_location=args.device)
        
        if var_config["time"]:
            current_meta = data["meta"]
            lats = current_meta[:, 0]
            alts = current_meta[:, 3]
            mask = (lats > 15) & (lats < 66) & (alts < 4000)
            X = data["latents"][mask]
            labels = current_meta[:, -1][mask].cpu().numpy().flatten()
        else:
            X = data["latents"]
            labels = data["meta"][:, var_config["meta_idx"]].cpu().numpy().flatten()
            if var_config["circular"]:
                labels = labels * np.pi / 180.0

        if len(df_geom) == 1:
            geom_row = df_geom.iloc[0]
        else:
            row_idx = max(0, iter_val // 2 - 1)
            if row_idx >= len(df_geom):
                row_idx = len(df_geom) - 1
            geom_row = df_geom.iloc[row_idx]

        R_m = torch.tensor(geom_row["rotation_matrix"], device=args.device, dtype=torch.float32)
        t_m = torch.tensor(geom_row["translation_vec"], device=args.device, dtype=torch.float32)
        l_m = torch.tensor(geom_row["lambda_vec"], device=args.device, dtype=torch.float32)

        with torch.no_grad():
            X_centered = X.to(args.device) + t_m
            X_rotated = torch.matmul(X_centered, R_m)
            top_2 = torch.argsort(l_m.abs(), descending=True)[:2]
            Ls_model = X_rotated[:, top_2].cpu().numpy()
            
            dim1, dim2 = top_2[0].item(), top_2[1].item()
            lambda1, lambda2 = l_m[top_2[0]].item(), l_m[top_2[1]].item()

        plot_data.append({
            "iteration": iter_val,
            "Ls_model": Ls_model,
            "labels": labels,
            "dims": (dim1, dim2),
            "lambdas": (lambda1, lambda2)
        })
        all_labels.append(labels)

    if not plot_data:
        raise ValueError("No valid iterations were loaded successfully.")

    global_labels = np.concatenate(all_labels)
    vmin, vmax = global_labels.min(), global_labels.max()
    cmap = "twilight" if var_config["circular"] else "viridis"

    num_plots = len(plot_data)
    ncols = 3
    nrows = math.ceil(num_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)

    last_scatter = None
    for idx, item in enumerate(plot_data):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        
        Ls = item["Ls_model"]
        lbls = item["labels"]
        d1, d2 = item["dims"]
        l1, l2 = item["lambdas"]

        last_scatter = ax.scatter(
            Ls[:, 0], Ls[:, 1],
            c=lbls, cmap=cmap, s=3, alpha=0.4,
            vmin=vmin, vmax=vmax
        )

        # if var_config["circular"]:
        #     center_opt, corr_opt, mrl_opt = find_optimal_center(Ls, lbls)
        #     ax.scatter(
        #         center_opt[0], center_opt[1],
        #         color="lime", marker="*", s=180, edgecolor="black",
        #         label=f"Center (r={corr_opt:.2f}, R={mrl_opt:.2f})",
        #         zorder=4
        #     )
        #     ax.legend(loc="upper right", fontsize=8)

        # Force axis limits to tightly match data bounds
        x_min, x_max = Ls[:, 0].min(), Ls[:, 0].max()
        y_min, y_max = Ls[:, 1].min(), Ls[:, 1].max()
        pad_x = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
        pad_y = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

        ax.set_title(
            f"2D Affine Subspace for Feature Interpretation\nActive Dimensions: [{d1}, {d2}]",
            fontsize=10
        )
        ax.set_xlabel(f"Manifold Axis 1 (Lambda: {l1:.4f})", fontsize=9)
        ax.set_ylabel(f"Manifold Axis 2 (Lambda: {l2:.4f})", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(num_plots, nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r, c])

    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_scatter, cax=cbar_ax)
    cbar.set_label(var_title, fontsize=11)

    out_prefix = os.path.join(args.output_dir, f"{var_code}_seed{args.seed}_panel")
    pdf_path = f"{out_prefix}.pdf"
    png_path = f"{out_prefix}.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Panels successfully generated:\n  - PDF: {pdf_path}\n  - PNG: {png_path}")

if __name__ == "__main__":
    main()