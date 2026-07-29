import matplotlib.pyplot as plt

def plot_violins(data: dict, title: str, out_path: str):
    keys = [k for k in ["moran", "circ", "mrl"] if f"{k}_model" in data]
    if not keys:
        return

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        m, b = data[f"{k}_model"], data[f"{k}_baseline"]
        parts = ax.violinplot([m, b], positions=[1, 2], showmeans=True, showmedians=True)
        for pc, color in zip(parts["bodies"], ["#4c78a8", "#f58518"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Model", "Baseline"])
        ax.set_ylabel(k.upper())
        ax.set_title(f"{k.upper()}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
