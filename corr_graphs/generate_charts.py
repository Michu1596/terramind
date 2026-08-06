#!/usr/bin/env python3
"""
Generate Moran's I and Circular Correlation charts with confidence intervals.

Usage examples:
    # Moran chart with standard CI, all models, with baseline
    python generate_charts.py --files results/L_results.csv results/T_results.csv \
        results/Long_results.csv results/H_results.csv --chart-type moran \
        --ci-method ci --include-baseline

    # Circular chart with violin plots, only Time and Longitude used
    python generate_charts.py --files resu lts/T_results.csv results/Long_results.csv \
        --chart-type circular --ci-method violin --include-baseline

    # Moran chart with bootstrap CI
    python generate_charts.py --files results/*.csv --chart-type moran \
        --ci-method bootstrap --confidence 0.99
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# prefix -> (display_name, color)
MODEL_CONFIG = {
    "L_": ("Latitude Model", "#1f77b4"),      # blue
    "T_": ("Time Model", "#2ca02c"),          # green
    "Long_": ("Longitude Model", "#d62728"),  # red
    "H_": ("Altitude Model", "#000000"),      # black
}

CIRCULAR_PREFIXES = {"T_", "Long_"}

# Sort by prefix length descending so "Long_" matches before "L_"
_SORTED_CONFIG = sorted(MODEL_CONFIG.items(), key=lambda x: len(x[0]), reverse=True)


def identify_model(filename: str):
    """Return (label, color) for a given filename based on prefix."""
    basename = Path(filename).name
    for prefix, (label, color) in _SORTED_CONFIG:
        if basename.startswith(prefix):
            return label, color
    raise ValueError(
        f"Cannot identify model for '{basename}'. "
        f"Expected prefix in: {list(MODEL_CONFIG.keys())}"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets(file_paths, chart_type):
    """Load CSVs and return {model_label: info_dict}."""
    data = {}
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"[WARN] File not found: {path}", file=sys.stderr)
            continue

        df = pd.read_csv(path)
        label, color = identify_model(path.name)

        # Circular charts only accept Time and Longitude
        if chart_type == "circular" and label not in {MODEL_CONFIG[p][0] for p in CIRCULAR_PREFIXES}:
            print(f"[INFO] Skipping '{label}' for circular chart.", file=sys.stderr)
            continue

        # Validate required columns
        if chart_type == "moran":
            required = {"iteration", "moran_I"}
        else:
            required = {"iteration", "circ_corr"}

        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] {path.name} missing columns {missing}, skipping.", file=sys.stderr)
            continue

        data[label] = {"df": df, "color": color, "file": path.name}

    return data


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_ci(values, method="ci", confidence=0.95, n_bootstrap=1000):
    """
    Compute mean and confidence interval.

    Returns:
        (mean, lower, upper)  where lower/upper may be None for violin method.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(values))

    if method == "violin":
        return mean, None, None

    if method == "ci":
        if n < 2:
            return mean, mean, mean
        std_err = np.std(values, ddof=1) / np.sqrt(n)
        # Normal approximation is fine for n >= 30; use t for small n
        if n >= 30:
            z = 1.96  # ~95% or scaled by confidence
            # More precise: norm.ppf
            from scipy.stats import norm
            z = norm.ppf((1 + confidence) / 2)
        else:
            from scipy.stats import t as t_dist
            z = t_dist.ppf((1 + confidence) / 2, n - 1)
        margin = z * std_err
        return mean, mean - margin, mean + margin

    if method == "bootstrap":
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=n, replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)
        alpha = (1 - confidence) / 2
        return mean, float(np.percentile(boot_means, alpha * 100)), float(
            np.percentile(boot_means, (1 - alpha) * 100)
        )

    raise ValueError(f"Unknown CI method: {method}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_chart(data_dict, chart_type, ci_method, include_baseline, output_dir, confidence=0.95):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Gather all iteration values
    all_iterations = set()
    for info in data_dict.values():
        all_iterations.update(info["df"]["iteration"].unique())
    all_iterations = sorted(all_iterations)

    # Offsets for violins so they don't overlap
    n_models = len(data_dict)
    violin_width = 0.35
    if n_models > 1:
        offsets = np.linspace(-(n_models - 1) * violin_width / 2, (n_models - 1) * violin_width / 2, n_models)
    else:
        offsets = np.array([0.0])

    baseline_handles = []
    baseline_labels = []

    for idx, (label, info) in enumerate(data_dict.items()):
        df = info["df"]
        color = info["color"]

        records = []
        for it in all_iterations:
            subset = df[df["iteration"] == it]

            if chart_type == "moran":
                vals = subset["moran_I"].values
                base_col = "base_moran_mean"
            else:
                vals = subset["circ_corr"].values
                base_col = "base_circ_mean"

            mean, lower, upper = mean_ci(vals, ci_method, confidence)

            base_vals = subset[base_col].values if base_col in subset.columns else np.array([])
            base_mean = float(np.mean(base_vals)) if len(base_vals) > 0 else np.nan

            records.append({
                "iteration": it,
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "values": vals,
                "base_mean": base_mean,
                "base_vals": base_vals,
            })

        agg = pd.DataFrame(records)
        x = agg["iteration"].values
        y = agg["mean"].values

        # Model line
        ax.plot(x, y, color=color, marker="o", markersize=5, label=label, zorder=3)

        # Confidence interval or violin
        if ci_method in ("ci", "bootstrap"):
            lowers = agg["lower"].values
            uppers = agg["upper"].values
            ax.fill_between(x, lowers, uppers, color=color, alpha=0.2, zorder=1)
        elif ci_method == "violin":
            positions = x + offsets[idx]
            datasets = [v for v in agg["values"].values if len(v) > 0]
            pos = [p for p, v in zip(positions, agg["values"].values) if len(v) > 0]
            if datasets:
                parts = ax.violinplot(
                    datasets, positions=pos, widths=violin_width * 0.8,
                    showmeans=False, showmedians=False, showextrema=False
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(0.35)

        # Baseline
        if include_baseline:
            base_y = agg["base_mean"].values
            if not np.all(np.isnan(base_y)):
                # Check if baseline has any variance across seeds
                has_variance = any(
                    len(r["base_vals"]) > 1 and np.std(r["base_vals"]) > 1e-12
                    for _, r in agg.iterrows()
                )

                (line,) = ax.plot(
                    x, base_y, color=color, linestyle="--", marker="s",
                    markersize=4, alpha=0.7, zorder=2
                )
                baseline_handles.append(line)
                baseline_labels.append(f"{label} (baseline)")

                if has_variance and ci_method in ("ci", "bootstrap"):
                    b_lowers, b_uppers = [], []
                    for _, r in agg.iterrows():
                        if len(r["base_vals"]) > 0:
                            _, bl, bu = mean_ci(r["base_vals"], ci_method, confidence)
                            b_lowers.append(bl)
                            b_uppers.append(bu)
                        else:
                            b_lowers.append(np.nan)
                            b_uppers.append(np.nan)
                    ax.fill_between(x, b_lowers, b_uppers, color=color, alpha=0.1, zorder=1)

    # Styling
    if chart_type == "moran":
        ax.set_ylabel("Moran's I")
        ax.set_title("Moran's I and depth (Model vs Baseline)")
        fname = "moran"
    else:
        ax.set_ylabel("Circular Correlation")
        ax.set_title("Circular Correlation and depth (Model vs Baseline)")
        fname = "circular"

    ax.set_xlabel("Depth")
    ax.set_xticks(all_iterations)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Legend: model lines first, then baseline lines
    handles, labels = ax.get_legend_handles_labels()
    if include_baseline and baseline_handles:
        handles.extend(baseline_handles)
        labels.extend(baseline_labels)
    ax.legend(handles, labels, loc="best", fontsize="small")

    plt.tight_layout()

    suffix = f"{ci_method}{'_baseline' if include_baseline else ''}"
    png_path = output_dir / f"{fname}_{suffix}.png"
    pdf_path = output_dir / f"{fname}_{suffix}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate charts with confidence intervals from seed-based results."
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="Input CSV files (e.g. results/L_results.csv)"
    )
    parser.add_argument(
        "--chart-type", choices=["moran", "circular"], required=True,
        help="Type of chart to generate"
    )
    parser.add_argument(
        "--ci-method", choices=["ci", "violin", "bootstrap"], default="ci",
        help=(
            "ci = mean with standard-error CI (t-dist for n<30, normal for n>=30); "
            "violin = violin plot at each depth; "
            "bootstrap = percentile bootstrap CI (1000 samples)"
        )
    )
    parser.add_argument(
        "--include-baseline", action="store_true",
        help="Include baseline curves (dashed) with CI when variance is available"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level (default 0.95)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="corr_graphs/results",
        help="Directory for output PNG and PDF files"
    )

    args = parser.parse_args()

    data_dict = load_datasets(args.files, args.chart_type)
    if not data_dict:
        print("[ERROR] No valid datasets loaded.", file=sys.stderr)
        sys.exit(1)

    plot_chart(
        data_dict,
        args.chart_type,
        args.ci_method,
        args.include_baseline,
        args.output_dir,
        args.confidence,
    )


if __name__ == "__main__":
    main()
