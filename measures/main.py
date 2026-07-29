import argparse
import ast
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from measures.config import TRANSFORMATIONS
from measures.stats import moran, optimal_center, bootstrap_ci
from measures.baselines import get_baseline
from measures.viz import plot_violins

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_transform_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("rotation_matrix", "translation_vec", "lambda_vec"):
        df[col] = df[col].apply(ast.literal_eval)
    return df


def run():
    parser = argparse.ArgumentParser(description="Evaluate learned spatial transformations")
    parser.add_argument("transformation", choices=["L", "Long", "H", "T"], help="Which transformation to evaluate")
    parser.add_argument("--transformations-dir", default="./transformations", help="Folder with seed CSVs")
    parser.add_argument("--data-dir", default="./data", help="Folder with latent .pt files")
    parser.add_argument("--n-baselines", type=int, default=100, help="Baseline runs per seed-iteration")
    parser.add_argument("--use-bbox", action="store_true", help="Use bbox center instead of Nelder-Mead for circular stats")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--output-dir", default="./results", help="Where to write outputs")
    args = parser.parse_args()

    cfg = TRANSFORMATIONS[args.transformation]
    csv_files = sorted(glob.glob(str(Path(args.transformations_dir) / cfg["pattern"])))
    if not csv_files:
        logger.error(f"No files matched {cfg['pattern']} in {args.transformations_dir}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} seed files for {cfg['name']}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_records = []
    baseline_records = []
    skipped = 0

    for fi, csv_path in enumerate(csv_files):
        seed_name = Path(csv_path).stem
        logger.info(f"[{fi + 1}/{len(csv_files)}] Processing {seed_name}")
        df = load_transform_csv(csv_path)

        for _, row in df.iterrows():
            iteration = int(row["idx"])
            pt_path = Path(args.data_dir) / f"latents_{iteration}_92160_for_intervention.pt"
            if not pt_path.exists():
                skipped += 1
                logger.warning(f"  Missing {pt_path.name} — skipping iteration {iteration}")
                continue

            data = torch.load(pt_path, map_location=args.device)
            X, meta = data["latents"], data["meta"]

            if cfg.get("time_filter"):
                mask = (meta[:, 0] > 15) & (meta[:, 0] < 66) & (meta[:, 3] < 4000)
                X, meta = X[mask], meta[mask]

            labels = meta[:, cfg["meta_idx"]].cpu().numpy().flatten()
            if "transform" in cfg:
                labels = cfg["transform"](labels)
            elif cfg.get("circular"):
                labels = labels * np.pi / 180

            if len(labels) == 0:
                continue

            X = X.to(args.device)
            N = X.shape[1]

            # ---- Model transformation ----
            R = torch.tensor(row["rotation_matrix"], device=args.device, dtype=torch.float32)
            t = torch.tensor(row["translation_vec"], device=args.device, dtype=torch.float32)
            l = torch.tensor(row["lambda_vec"], device=args.device, dtype=torch.float32)

            with torch.no_grad():
                Xm = torch.matmul(X + t, R)
                top2 = torch.argsort(l.abs(), descending=True)[:2]
                Lm = Xm[:, top2].cpu().numpy()

            mi, p_val, z_val = moran(Lm, labels)
            rec = {
                "seed": seed_name,
                "iteration": iteration,
                "moran_I": mi,
                "moran_p": p_val,
                "moran_z": z_val,
            }

            if cfg.get("circular"):
                _, cc, mr = optimal_center(Lm, labels, use_bbox=args.use_bbox)
                rec["circ_corr"] = cc
                rec["mrl"] = mr

            # ---- Baselines ----
            b_morans, b_corrs, b_mrls = [], [], []
            for _ in range(args.n_baselines):
                b = get_baseline(N, args.device)
                with torch.no_grad():
                    Xb = torch.matmul(X + b["t"], b["R"])
                    Lb = Xb[:, :2].cpu().numpy()
                bmi, _, _ = moran(Lb, labels)
                b_morans.append(bmi)
                if cfg.get("circular"):
                    _, bcc, bmr = optimal_center(Lb, labels, use_bbox=args.use_bbox)
                    b_corrs.append(bcc)
                    b_mrls.append(bmr)

            rec["base_moran_mean"] = float(np.mean(b_morans))
            base_rec = {"seed": seed_name, "iteration": iteration, "moran": b_morans}

            if cfg.get("circular"):
                rec["base_circ_mean"] = float(np.mean(b_corrs))
                rec["base_mrl_mean"] = float(np.mean(b_mrls))
                base_rec["circ"] = b_corrs
                base_rec["mrl"] = b_mrls

            model_records.append(rec)
            baseline_records.append(base_rec)

    if not model_records:
        logger.error("No results were produced.")
        sys.exit(1)

    df_model = pd.DataFrame(model_records)
    df_model.to_csv(out_dir / f"{args.transformation}_results.csv", index=False)

    # ---- Violin data ----
    vdata = {
        "moran_model": df_model["moran_I"].tolist(),
        "moran_baseline": [v for b in baseline_records for v in b["moran"]],
    }
    if cfg.get("circular"):
        vdata["circ_model"] = df_model["circ_corr"].tolist()
        vdata["circ_baseline"] = [v for b in baseline_records for v in b["circ"]]
        vdata["mrl_model"] = df_model["mrl"].tolist()
        vdata["mrl_baseline"] = [v for b in baseline_records for v in b["mrl"]]

    plot_violins(vdata, cfg["name"], str(out_dir / f"{args.transformation}_violins.png"))

    # ---- Summary stats ----
    summary = {"transformation": args.transformation, "seeds": len(csv_files), "skipped_iterations": skipped}
    logger.info("\n=== SUMMARY ===")

    for col, pretty in [("moran_I", "Moran\'s I"), ("circ_corr", "Circular Corr"), ("mrl", "MRL")]:
        if col not in df_model.columns:
            continue
        vals = df_model[col].dropna()
        lo, hi = bootstrap_ci(vals)
        summary[f"{col}_mean"] = round(float(vals.mean()), 4)
        summary[f"{col}_ci95"] = [round(lo, 4), round(hi, 4)]
        logger.info(f"{pretty}: mean={vals.mean():.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")

    # ---- Wilcoxon tests (model vs mean baseline, paired) ----
    w, p = wilcoxon(df_model["moran_I"], df_model["base_moran_mean"], alternative="greater")
    summary["moran_wilcoxon_W"] = round(w, 2)
    summary["moran_wilcoxon_p"] = round(p, 4)
    logger.info(f"Moran\'s I Wilcoxon: W={w:.2f}, p={p:.4f}")

    if cfg.get("circular"):
        w, p = wilcoxon(df_model["circ_corr"], df_model["base_circ_mean"], alternative="greater")
        summary["circ_wilcoxon_W"] = round(w, 2)
        summary["circ_wilcoxon_p"] = round(p, 4)
        logger.info(f"Circ Corr Wilcoxon: W={w:.2f}, p={p:.4f}")

        w, p = wilcoxon(df_model["mrl"], df_model["base_mrl_mean"], alternative="greater")
        summary["mrl_wilcoxon_W"] = round(w, 2)
        summary["mrl_wilcoxon_p"] = round(p, 4)
        logger.info(f"MRL Wilcoxon: W={w:.2f}, p={p:.4f}")

    if skipped:
        logger.info(f"Skipped {skipped} missing iteration(s) total.")

    with open(out_dir / f"{args.transformation}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done. Outputs written to {out_dir}")


if __name__ == "__main__":
    run()
