"""
Visualization — Publication-quality plots for the self-pruning network.

Generates four key figures:
1. Gate distribution histogram (bimodal spike at 0 and cluster near 1)
2. Layer-wise gate value heatmaps
3. Lambda trade-off curve (sparsity vs accuracy, Pareto frontier)
4. Training curves (loss, accuracy, sparsity over epochs)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for server/CI environments
matplotlib.use("Agg")

# Style configuration for publication quality
plt.rcParams.update(
    {
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560",
        "axes.labelcolor": "#eee",
        "text.color": "#eee",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 150,
    }
)

# Color palette — vibrant, distinguishable
LAYER_COLORS = ["#e94560", "#0f3460", "#53d8fb", "#f5a623", "#7F77DD"]
ACCENT = "#e94560"
BG_DARK = "#1a1a2e"


def plot_gate_distribution(
    gate_values: dict,
    threshold: float = 0.01,
    save_path: str = "outputs/gate_distribution.png",
    title: str = "Gate Value Distribution Across Layers",
):
    """Plot overlaid histograms of gate values for each prunable layer.

    Shows the characteristic bimodal distribution: a spike near 0
    (pruned weights) and a cluster near 1 (active weights).

    Args:
        gate_values: Dict mapping layer names to gate value tensors.
        threshold: Pruning threshold (drawn as vertical line).
        save_path: Output file path.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, gates) in enumerate(gate_values.items()):
        values = gates.numpy().flatten() if hasattr(gates, "numpy") else np.array(gates).flatten()
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        short_name = name.split(".")[-1] if "." in name else name
        ax.hist(
            values,
            bins=60,
            range=(0, 1),
            alpha=0.55,
            color=color,
            label=short_name,
            edgecolor=color,
            linewidth=0.5,
        )

    # Threshold line
    ax.axvline(
        threshold,
        color="#ff6b6b",
        linestyle="--",
        linewidth=1.5,
        label=f"prune threshold ({threshold})",
        alpha=0.9,
    )

    ax.set_xlabel("Gate Value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.7, loc="upper center")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_layer_heatmaps(
    gate_values: dict,
    save_path: str = "outputs/layer_heatmaps.png",
    title: str = "Gate Values by Layer (Heatmap)",
):
    """Plot gate values as 2D heatmaps for each prunable layer.

    Reshapes the gate tensor into its natural (out_features, in_features)
    shape and displays as an image.

    Args:
        gate_values: Dict mapping layer names to gate value tensors.
        save_path: Output file path.
        title: Overall figure title.
    """
    n_layers = len(gate_values)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    for i, (name, gates) in enumerate(gate_values.items()):
        values = gates.numpy() if hasattr(gates, "numpy") else np.array(gates)
        if values.ndim == 1:
            # Reshape to square-ish
            side = int(np.ceil(np.sqrt(values.shape[0])))
            padded = np.zeros(side * side)
            padded[: values.shape[0]] = values
            values = padded.reshape(side, side)

        short_name = name.split(".")[-1] if "." in name else name
        im = axes[i].imshow(values, cmap="inferno", vmin=0, vmax=1, aspect="auto")
        axes[i].set_title(short_name, fontsize=11, color="#eee")
        axes[i].set_xlabel("Input neurons")
        axes[i].set_ylabel("Output neurons")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Gate value")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#eee")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_lambda_tradeoff(
    results: list[dict],
    save_path: str = "outputs/lambda_tradeoff.png",
    title: str = "Sparsity vs Accuracy Trade-off",
):
    """Plot the accuracy-sparsity Pareto frontier across experiments.

    Each point is one experiment (different λ config). Points on the
    Pareto frontier are connected with a line.

    Args:
        results: List of dicts, each with keys:
            'name', 'test_accuracy', 'sparsity', 'lambda_max'
        save_path: Output file path.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    sparsities = [r["sparsity"] * 100 for r in results]
    accuracies = [r["test_accuracy"] * 100 for r in results]
    names = [r["name"] for r in results]

    # Scatter points
    scatter = ax.scatter(
        sparsities,
        accuracies,
        c=LAYER_COLORS[: len(results)],
        s=120,
        zorder=5,
        edgecolors="white",
        linewidth=1.5,
    )

    # Labels
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (sparsities[i], accuracies[i]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="#ccc",
            fontweight="bold",
        )

    # Pareto frontier: sort by sparsity, keep non-dominated
    sorted_idx = sorted(range(len(results)), key=lambda i: sparsities[i])
    pareto_x, pareto_y = [], []
    best_acc = -1
    for idx in sorted_idx:
        if accuracies[idx] >= best_acc:
            pareto_x.append(sparsities[idx])
            pareto_y.append(accuracies[idx])
            best_acc = accuracies[idx]

    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, "--", color=ACCENT, alpha=0.6, linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(
    histories: dict[str, dict],
    save_path: str = "outputs/training_curves.png",
    title: str = "Training Curves",
):
    """Plot training metrics over epochs for multiple experiments.

    Creates a 2×2 grid: classification loss, train accuracy,
    validation accuracy, sparsity.

    Args:
        histories: Dict mapping config name → history dict
            (with keys: cls_loss, train_acc, val_acc, sparsity).
        save_path: Output file path.
        title: Overall figure title.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("cls_loss", "Classification Loss", axes[0, 0]),
        ("train_acc", "Train Accuracy", axes[0, 1]),
        ("val_acc", "Validation Accuracy", axes[1, 0]),
        ("sparsity", "Sparsity", axes[1, 1]),
    ]

    for metric_key, metric_title, ax in metrics:
        for i, (name, history) in enumerate(histories.items()):
            if metric_key in history and history[metric_key]:
                values = history[metric_key]
                color = LAYER_COLORS[i % len(LAYER_COLORS)]
                ax.plot(values, label=name, color=color, linewidth=1.5, alpha=0.85)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_title)
        ax.set_title(metric_title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")
