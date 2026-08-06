#!/usr/bin/env python3
"""
Generate Moran's I and Circular Correlation charts with error bars and JSON export.

Usage examples:
    # Moran chart with standard-error CI error bars, all models, with baseline
    python generate_charts.py --files results/L_results.csv results/T_results.csv \
        results/Long_results.csv results/H_results.csv --chart-type moran \
        --ci-method ci --include-baseline

    # Circular chart with bootstrap CI, show min/max range whiskers
    python generate_charts.py --files results/T_results.csv results/Long_results.csv \
        --chart-type circular --ci-method bootstrap --include-baseline --show-range

    # Violin plots (no error bars, but JSON still contains all stats)
    python generate_charts.py --files results/*.csv --chart-type moran \
        --ci-method violin --include-baseline
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "L_": ("Latitude Model", "#1f77b4"),      # blue
    "T_": ("Time Model", "#2ca02c"),          # green
    "Long_": ("Longitude Model", "#d62728"),  # red
    "H_": ("Altitude Model", "#000000"),      # black
}

CIRCULAR_PREFIXES = {"T_", "Long_"}
_SORTED_CONFIG = sorted(MODEL_CONFIG.items(), key=lambda x: len(x[0]), reverse=True)


def identify_model(filename: str):
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
    data = {}
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"[WARN] File not found: {path}", file=sys.stderr)
            continue

        df = pd.read_csv(path)
        label, color = identify_model(path.name)

        if chart_type == "circular" and label not in {MODEL_CONFIG[p][0] for p in CIRCULAR_PREFIXES}:
            print(f"[INFO] Skipping '{label}' for circular chart.", file=sys.stderr)
            continue

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

def compute_wilcoxon(values, baseline_vals):
    """
    Compute Wilcoxon test of model values against baseline.
    If baseline is constant -> one-sample test (values - constant).
    If baseline varies -> paired test (values, baseline_vals).
    Returns dict or None if insufficient data.
    """
    values = np.asarray(values, dtype=float)
    baseline_vals = np.asarray(baseline_vals, dtype=float)
    values = values[~np.isnan(values)]
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]

    if len(values) == 0:
        return None

    if len(baseline_vals) == 0:
        return None
    baseline_is_constant = np.allclose(baseline_vals, baseline_vals[0])

    try:
        if baseline_is_constant:
            diff = values - baseline_vals[0]
            diff = diff[diff != 0]
            if len(diff) == 0:
                return {"statistic": 0.0, "pvalue": 1.0, "baseline_is_constant": True}
            if len(diff) < 2:
                return {"statistic": None, "pvalue": None, "baseline_is_constant": True,
                        "note": "insufficient non-zero differences"}
            w, p = stats.wilcoxon(diff, alternative="two-sided")
            return {"statistic": float(w), "pvalue": float(p), "baseline_is_constant": True}
        else:
            n = min(len(values), len(baseline_vals))
            if n < 2:
                return None
            w, p = stats.wilcoxon(values[:n], baseline_vals[:n], alternative="two-sided")
            return {"statistic": float(w), "pvalue": float(p), "baseline_is_constant": False}
    except Exception as e:
        return {"statistic": None, "pvalue": None, "baseline_is_constant": baseline_is_constant, "error": str(e)}


def compute_stats(values, baseline_vals, method="ci", confidence=0.95, n_bootstrap=1000):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    baseline_vals = np.asarray(baseline_vals, dtype=float)
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]

    result = {
        "n": n,
        "mean": float(np.mean(values)) if n > 0 else None,
        "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(values)) if n > 0 else None,
        "max": float(np.max(values)) if n > 0 else None,
        "ci_lower": None,
        "ci_upper": None,
    }

    if n == 0:
        return result, compute_wilcoxon(values, baseline_vals)

    mean = result["mean"]

    if method == "violin":
        result["ci_lower"] = float(np.percentile(values, 2.5))
        result["ci_upper"] = float(np.percentile(values, 97.5))
    elif method == "ci":
        if n >= 2:
            std_err = np.std(values, ddof=1) / np.sqrt(n)
            if n >= 30:
                z = stats.norm.ppf((1 + confidence) / 2)
            else:
                z = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = z * std_err
            result["ci_lower"] = mean - margin
            result["ci_upper"] = mean + margin
        else:
            result["ci_lower"] = mean
            result["ci_upper"] = mean
    elif method == "bootstrap":
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=n, replace=True)
            boot_means.append(np.mean(sample))
        boot_means = np.array(boot_means)
        alpha = (1 - confidence) / 2
        result["ci_lower"] = float(np.percentile(boot_means, alpha * 100))
        result["ci_upper"] = float(np.percentile(boot_means, (1 - alpha) * 100))
    else:
        raise ValueError(f"Unknown method: {method}")

    wilcoxon = compute_wilcoxon(values, baseline_vals)
    return result, wilcoxon


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_chart(data_dict, chart_type, ci_method, include_baseline, show_range, output_dir, confidence=0.95):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    all_iterations = set()
    for info in data_dict.values():
        all_iterations.update(info["df"]["iteration"].unique())
    all_iterations = sorted(all_iterations)

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

            base_vals = subset[base_col].values if base_col in subset.columns else np.array([])

            model_stats, wilcoxon = compute_stats(vals, base_vals, ci_method, confidence)
            base_stats, _ = compute_stats(base_vals, base_vals, ci_method, confidence)

            records.append({
                "iteration": it,
                "model_stats": model_stats,
                "baseline_stats": base_stats,
                "wilcoxon": wilcoxon,
                "values": vals,
                "base_vals": base_vals,
            })

        agg = pd.DataFrame(records)
        x = agg["iteration"].values
        y = np.array([r["model_stats"]["mean"] for r in records])

        # --- Model line with error bars ---
        if ci_method == "violin":
            ax.plot(x, y, color=color, marker="o", markersize=6, label=label, zorder=3, linewidth=2)
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
        else:
            # CI error bars
            ci_low = np.array([r["model_stats"]["ci_lower"] for r in records])
            ci_high = np.array([r["model_stats"]["ci_upper"] for r in records])
            yerr_lower = y - ci_low
            yerr_upper = ci_high - y
            yerr = np.vstack([yerr_lower, yerr_upper])

            ax.errorbar(
                x, y, yerr=yerr, color=color, marker="o", markersize=6,
                capsize=4, capthick=1.5, elinewidth=1.5, label=label, zorder=3
            )

            # Range whiskers (min/max)
            if show_range:
                mins = np.array([r["model_stats"]["min"] for r in records])
                maxs = np.array([r["model_stats"]["max"] for r in records])
                yerr_range_lower = y - mins
                yerr_range_upper = maxs - y
                yerr_range = np.vstack([yerr_range_lower, yerr_range_upper])
                ax.errorbar(
                    x, y, yerr=yerr_range, color=color, marker="", linestyle="",
                    capsize=2, capthick=1.0, elinewidth=1.0, alpha=0.5, zorder=2
                )

        # --- Baseline ---
        if include_baseline:
            base_y = np.array([r["baseline_stats"]["mean"] for r in records])
            if not np.all(np.isnan(base_y)):
                has_variance = any(
                    len(r["base_vals"]) > 1 and np.std(r["base_vals"]) > 1e-12
                    for _, r in agg.iterrows()
                )

                (line,) = ax.plot(
                    x, base_y, color=color, linestyle="--", marker="s",
                    markersize=5, alpha=0.7, zorder=2
                )
                baseline_handles.append(line)
                baseline_labels.append(f"{label} (baseline)")

                if has_variance and ci_method != "violin":
                    b_ci_low = np.array([r["baseline_stats"]["ci_lower"] for r in records])
                    b_ci_high = np.array([r["baseline_stats"]["ci_upper"] for r in records])
                    b_yerr_lower = base_y - b_ci_low
                    b_yerr_upper = b_ci_high - base_y
                    b_yerr = np.vstack([b_yerr_lower, b_yerr_upper])
                    ax.errorbar(
                        x, base_y, yerr=b_yerr, color=color, marker="", linestyle="",
                        capsize=3, capthick=1.0, elinewidth=1.0, alpha=0.4, zorder=1
                    )

                    if show_range:
                        b_mins = np.array([r["baseline_stats"]["min"] for r in records])
                        b_maxs = np.array([r["baseline_stats"]["max"] for r in records])
                        b_yerr_range_lower = base_y - b_mins
                        b_yerr_range_upper = b_maxs - base_y
                        b_yerr_range = np.vstack([b_yerr_range_lower, b_yerr_range_upper])
                        ax.errorbar(
                            x, base_y, yerr=b_yerr_range, color=color, marker="", linestyle="",
                            capsize=1.5, capthick=0.8, elinewidth=0.8, alpha=0.3, zorder=1
                        )

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

    handles, labels = ax.get_legend_handles_labels()
    if include_baseline and baseline_handles:
        handles.extend(baseline_handles)
        labels.extend(baseline_labels)
    ax.legend(handles, labels, loc="best", fontsize="small")

    plt.tight_layout()

    suffix = f"{ci_method}{'_range' if show_range else ''}{'_baseline' if include_baseline else ''}"
    png_path = output_dir / f"{fname}_{suffix}.png"
    pdf_path = output_dir / f"{fname}_{suffix}.pdf"

    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    return records, fname, suffix


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(data_dict, chart_type, ci_method, include_baseline, show_range, confidence, output_dir, fname, suffix):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "chart_type": chart_type,
            "ci_method": ci_method,
            "confidence": confidence,
            "include_baseline": include_baseline,
            "show_range": show_range,
        },
        "models": {},
    }

    for label, info in data_dict.items():
        df = info["df"]
        model_data = []

        if chart_type == "moran":
            val_col = "moran_I"
            base_col = "base_moran_mean"
        else:
            val_col = "circ_corr"
            base_col = "base_circ_mean"

        iterations = sorted(df["iteration"].unique())
        for it in iterations:
            subset = df[df["iteration"] == it]
            vals = subset[val_col].values
            base_vals = subset[base_col].values if base_col in subset.columns else np.array([])

            model_stats, wilcoxon = compute_stats(vals, base_vals, ci_method, confidence)
            base_stats, _ = compute_stats(base_vals, base_vals, ci_method, confidence)

            entry = {
                "depth": int(it),
                "model": {k: v for k, v in model_stats.items()},
                "baseline": {k: v for k, v in base_stats.items()},
            }
            if wilcoxon is not None:
                entry["wilcoxon"] = wilcoxon
            model_data.append(entry)

        result["models"][label] = {"data": model_data}

    json_path = output_dir / f"{fname}_{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate charts with error bars and JSON statistics from seed-based results."
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
            "ci = mean with standard-error CI error bars (t-dist for n<30, normal for n>=30); "
            "violin = violin plot at each depth; "
            "bootstrap = percentile bootstrap CI error bars (1000 samples)"
        )
    )
    parser.add_argument(
        "--include-baseline", action="store_true",
        help="Include baseline curves (dashed) with error bars when variance is available"
    )
    parser.add_argument(
        "--show-range", action="store_true",
        help="Add thin min/max whiskers alongside CI error bars"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level for CI and bootstrap (default 0.95)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="corr_graphs/results",
        help="Directory for output PNG, PDF, and JSON files"
    )

    args = parser.parse_args()

    data_dict = load_datasets(args.files, args.chart_type)
    if not data_dict:
        print("[ERROR] No valid datasets loaded.", file=sys.stderr)
        sys.exit(1)

    _, fname, suffix = plot_chart(
        data_dict,
        args.chart_type,
        args.ci_method,
        args.include_baseline,
        args.show_range,
        args.output_dir,
        args.confidence,
    )

    export_json(
        data_dict,
        args.chart_type,
        args.ci_method,
        args.include_baseline,
        args.show_range,
        args.confidence,
        args.output_dir,
        fname,
        suffix,
    )


if __name__ == "__main__":
    main()
