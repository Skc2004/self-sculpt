"""
Animated gate evolution — Shows how gate value distribution changes
over training epochs as a GIF animation.

This is a powerful visualization that demonstrates the pruning process:
gates start spread around 0.62 (sigmoid(0.5)) and progressively collapse
to a bimodal distribution with spikes at 0 (pruned) and near 1 (active).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

matplotlib.use("Agg")


def animate_gate_evolution(
    gate_history: list[np.ndarray],
    save_path: str = "outputs/gate_evolution.gif",
    fps: int = 8,
    interval: int = 150,
):
    """Create an animated GIF of gate value distribution evolving over epochs.

    Args:
        gate_history: List of 1D numpy arrays, one per epoch,
            containing all gate values at that epoch.
        save_path: Output GIF file path.
        fps: Frames per second for the GIF.
        interval: Milliseconds between frames.
    """
    if not gate_history:
        print("  No gate history to animate.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    def update(frame):
        ax.clear()

        ax.hist(
            gate_history[frame],
            bins=60,
            range=(0, 1),
            color="#7F77DD",
            alpha=0.75,
            edgecolor="#5c55b3",
            linewidth=0.3,
        )

        # Threshold line
        ax.axvline(
            0.01,
            color="#e94560",
            linestyle="--",
            linewidth=1.5,
            label="prune threshold",
            alpha=0.9,
        )

        # Sparsity annotation
        sparsity = (gate_history[frame] < 0.01).mean() * 100
        ax.text(
            0.98,
            0.95,
            f"Sparsity: {sparsity:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            color="#53d8fb",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.8),
        )

        ax.set_xlabel("Gate Value", fontsize=12, color="#eee")
        ax.set_ylabel("Count", fontsize=12, color="#eee")
        ax.set_title(
            f"Gate Distribution — Epoch {frame + 1}/{len(gate_history)}",
            fontsize=13,
            fontweight="bold",
            color="#eee",
        )
        ax.set_xlim(0, 1)
        ax.tick_params(colors="#aaa")
        ax.spines["bottom"].set_color("#e94560")
        ax.spines["left"].set_color("#e94560")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            loc="upper center",
            framealpha=0.5,
            fontsize=9,
            labelcolor="#eee",
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(gate_history),
        interval=interval,
        repeat=True,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved animated GIF: {save_path}")
