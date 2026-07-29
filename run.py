#!/usr/bin/env python3
"""
Train GeometryExtractor locally.

Example:
    python run.py --property H --seed 42 --output-id seed42
    python run.py --property L --seed 42 --output-id seed42
    python run.py --property Long --seed 42 --output-id seed42
    python run.py --property T --seed 42 --output-id seed42
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from model import GeometryExtractor
from train import geometry_extraction, set_seed

# ---------------------------------------------------------------------------
# Column mapping derived from the filtering cell:
#   lats = saved_meta[:, 0]
#   alts = saved_meta[:, 3]
#   times = saved_meta[:, -1]
# Longitude is assumed to be column 1 — verify if your meta layout differs.
# ---------------------------------------------------------------------------
PROPERTY_MAP = {
    "H":    {"col": 3,  "prefix": "H_geometry_result"},  # altitude
    "L":    {"col": 0,  "prefix": "L_geom"},              # latitude
    "Long": {"col": 1,  "prefix": "Long_geom"},           # longitude (assumed)
    "T":    {"col": -1, "prefix": "T_geom"},              # time
}


def parse_args():
    p = argparse.ArgumentParser(description="Geometry Extractor training")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--property",
        type=str,
        default="H",
        choices=list(PROPERTY_MAP.keys()),
        help="Which geo-property to predict",
    )
    p.add_argument(
        "--latent-idx",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 10, 12],
        help="Indices of latent files to load",
    )
    p.add_argument(
        "--output-id",
        type=str,
        default="001",
        help="Suffix for output filenames",
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="transformations")
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--target-lambda", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    prop = PROPERTY_MAP[args.property]
    meta_col = prop["col"]
    prefix = prop["prefix"]

    R_mats, t_vecs, lambda_vecs = [], [], []
    records = []

    for idx in args.latent_idx:
        pt_path = os.path.join(args.data_dir, f"latents_{idx}_92160_for_intervention.pt")
        if not os.path.exists(pt_path):
            print(f"[SKIP] not found: {pt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Loading {pt_path}  |  property={args.property} (meta col {meta_col})")
        print(f"{'='*60}")

        data = torch.load(pt_path, map_location=device)
        saved_latents = data["latents"]
        saved_meta = data["meta"]

        # ---- apply the same geographic filter as the original notebook ----
        lats = saved_meta[:, 0]
        alts = saved_meta[:, 3]
        mask = (lats > 15) & (lats < 66) & (alts < 4000)

        X = saved_latents[mask]
        y = saved_meta[mask, meta_col]
        if args.property == "Long":
            y = torch.pi * (y + 180) / 180

        if X.shape[0] == 0:
            print(f"[SKIP] no samples left after filtering for idx={idx}")
            continue

        # ---- retry loop -------------------------------------------------
        epsilon = 1e-1
        attempt = 1
        max_attempts = 3
        while True:
            model, metrics = geometry_extraction(
                X,
                y,
                target_lambda=args.target_lambda,
                verbose=2,
                alpha_coeff=1.35,
                beta_coeff=1.45,
                n_epochs=args.epochs,
                device=device,
                seed=args.seed + attempt - 1,
            )

            nonzero = int((model.lambda_vec.abs() > epsilon).sum().item())
            R = model.rotation_matrix
            ortho_err = float(
                torch.norm(
                    torch.matmul(R.t(), R) - torch.eye(R.size(1), device=R.device),
                    p=2,
                ).item()
            )

            if nonzero <= args.target_lambda and ortho_err < epsilon:
                break
            elif attempt >= max_attempts:
                print(f"RED FLAG for idx={idx}")
                print(f"Final: nonzero={nonzero}, ortho_err={ortho_err:.6f}")
                break

            print(f"Retrying idx={idx}: nonzero={nonzero}, ortho_err={ortho_err:.4f}")
            attempt += 1

        # ---- save weights ------------------------------------------------
        model_name = f"{prefix}_{args.output_id}_idx{idx}.pt"
        model_path = os.path.join(args.model_dir, model_name)
        torch.save(model.state_dict(), model_path)
        print(f"[SAVED] model -> {model_path}")

        # ---- collect transforms ------------------------------------------
        R_mats.append(model.rotation_matrix.detach().cpu())
        t_vecs.append(model.t_vec.detach().cpu())
        lambda_vecs.append(model.lambda_vec.detach().cpu())

        records.append(
            {
                "idx": idx,
                "seed": args.seed,
                "mae": metrics[0],
                "nonzero_count": metrics[1],
                "det": metrics[2],
                "orthonorm": metrics[3],
                "retries": attempt,
            }
        )

    # ---- save CSV -------------------------------------------------------
    if R_mats:
        df = pd.DataFrame(
            {
                "idx": args.latent_idx[: len(R_mats)],
                "rotation_matrix": [r.tolist() for r in R_mats],
                "translation_vec": [t.tolist() for t in t_vecs],
                "lambda_vec": [l.tolist() for l in lambda_vecs],
            }
        )
        csv_name = f"{prefix}_{args.output_id}.csv"
        csv_path = os.path.join(args.output_dir, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"\n[SAVED] transforms -> {csv_path}")

        metrics_name = f"{prefix}_{args.output_id}_metrics.json"
        metrics_path = os.path.join(args.output_dir, metrics_name)
        with open(metrics_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[SAVED] metrics -> {metrics_path}")


if __name__ == "__main__":
    main()