"""
run_all.py — Single entry point to run all experiments.

Usage:
    python experiments/run_all.py              # full run (6 configs, 50 epochs each)
    python experiments/run_all.py --quick      # smoke test (2 configs, 5 epochs)
    python experiments/run_all.py --epochs 30  # custom epoch count

This script:
1. Downloads CIFAR-10 (auto-cached)
2. Trains a baseline model (no pruning)
3. Trains PrunableNet with each λ/schedule config
4. Evaluates all models and generates comparison report
5. Creates all visualizations (plots + animated GIF)
6. Saves results to outputs/results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.losses.sparsity_loss import SparsityLoss
from src.models.baseline_net import BaselineNet
from src.models.prunable_net import PrunableNet
from src.training.lambda_scheduler import LambdaScheduler
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    evaluate_accuracy,
    count_flops,
    compute_flops_reduction,
    measure_latency,
    full_evaluation_report,
)
from src.evaluation.diagnostics import verify_gradient_flow, get_gate_statistics
from src.visualization.plots import (
    plot_gate_distribution,
    plot_layer_heatmaps,
    plot_lambda_tradeoff,
    plot_training_curves,
)
from src.visualization.animate import animate_gate_evolution


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        print(f"  Using CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("  Using CPU")
    return "cpu"


def get_data_loaders(batch_size: int = 128, num_workers: int = 2):
    """Create CIFAR-10 train/test data loaders.

    Uses standard normalization for CIFAR-10.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    train_set = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def load_configs(config_dir: Path, quick: bool = False) -> list[dict]:
    """Load all YAML experiment configs.

    Args:
        config_dir: Path to configs/ directory.
        quick: If True, load only 2 configs for smoke testing.

    Returns:
        List of config dicts.
    """
    configs = []
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
            config["name"] = yaml_file.stem
            configs.append(config)

    if quick:
        # Pick one static and one annealed for smoke test
        static = [c for c in configs if c.get("schedule_mode") == "static"]
        annealed = [c for c in configs if c.get("schedule_mode") != "static"]
        configs = (static[:1] if static else []) + (annealed[:1] if annealed else [])
        if not configs and len(configs) > 0:
            configs = configs[:2]

    return configs


def train_baseline(train_loader, test_loader, device, epochs, output_dir):
    """Train the unpruned baseline model."""
    print("\n" + "=" * 70)
    print("BASELINE (no pruning)")
    print("=" * 70)

    model = BaselineNet()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lambda_sched=None,
        device=device,
        sparsity_loss=None,
        checkpoint_dir=str(output_dir),
    )

    history = trainer.fit(train_loader, test_loader, epochs=epochs)

    # Rename checkpoint
    src = output_dir / "best_model.pth"
    dst = output_dir / "baseline_best.pth"
    if src.exists():
        src.rename(dst)

    test_acc = evaluate_accuracy(model.to(device), test_loader, device)
    print(f"  Baseline test accuracy: {test_acc:.4f}")

    return model, history, test_acc


def train_prunable(config, train_loader, test_loader, device, epochs, output_dir):
    """Train a prunable model with a specific λ config."""
    name = config["name"]
    lambda_max = config["lambda_max"]
    schedule_mode = config.get("schedule_mode", "static")
    warmup_epochs = config.get("warmup_epochs", 5)
    gate_strategy = config.get("gate_strategy", "sigmoid")
    sparsity_norm = config.get("sparsity_norm", "L1")
    lr = config.get("learning_rate", 1e-3)

    print(f"\n{'=' * 70}")
    print(f"CONFIG: {name} | lambda={lambda_max} | schedule={schedule_mode} | gate={gate_strategy}")
    print(f"{'=' * 70}")

    model = PrunableNet(gate_strategy=gate_strategy)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    lambda_sched = LambdaScheduler(
        lambda_max=lambda_max,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        mode=schedule_mode,
    )
    sparsity_loss = SparsityLoss(norm=sparsity_norm)

    # Config-specific output dir
    config_dir = output_dir / name
    config_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lambda_sched=lambda_sched,
        device=device,
        sparsity_loss=sparsity_loss,
        checkpoint_dir=str(config_dir),
    )

    # Verify gradient flow on first batch
    grad_norms = verify_gradient_flow(model.to(device))
    x_sample, y_sample = next(iter(train_loader))
    x_sample, y_sample = x_sample[:4].to(device), y_sample[:4].to(device)
    logits = model(x_sample)
    loss = nn.functional.cross_entropy(logits, y_sample) + lambda_max * sparsity_loss(model)
    loss.backward()
    print(f"  Gradient flow check: {grad_norms}")
    optimizer.zero_grad()

    history = trainer.fit(train_loader, test_loader, epochs=epochs)

    # Copy best model to main outputs
    best_src = config_dir / "best_model.pth"
    if best_src.exists():
        import shutil
        shutil.copy2(best_src, output_dir / f"{name}_best.pth")

    test_acc = evaluate_accuracy(model.to(device), test_loader, device)
    sparsity_report = model.get_sparsity_report()

    print(f"\n  Results for {name}:")
    print(f"    Test accuracy: {test_acc:.4f}")
    print(f"    Sparsity: {sparsity_report}")

    # Gate statistics
    gate_stats = get_gate_statistics(model)
    for layer_name, stats in gate_stats.items():
        print(
            f"    {layer_name}: mean={stats['mean']:.3f} "
            f"near_0={stats['pct_near_zero']:.1%} "
            f"near_1={stats['pct_near_one']:.1%}"
        )

    return model, history, test_acc, sparsity_report, trainer.gate_history


def main():
    parser = argparse.ArgumentParser(description="Run self-pruning NN experiments")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (2 configs, 5 epochs)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()

    epochs = args.epochs or (5 if args.quick else 50)
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    config_dir = PROJECT_ROOT / "experiments" / "configs"

    print("=" * 70)
    print("SELF-PRUNING NEURAL NETWORK — EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Epochs: {epochs}")
    print(f"  Quick mode: {args.quick}")

    device = get_device()
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size, num_workers=args.workers
    )

    # Load configs
    configs = load_configs(config_dir, quick=args.quick)
    print(f"  Loaded {len(configs)} experiment configs")

    # ---- Train baseline ----
    baseline_model, baseline_history, baseline_acc = train_baseline(
        train_loader, test_loader, device, epochs, output_dir
    )

    # ---- Train prunable variants ----
    all_results = []
    all_histories = {"baseline": baseline_history}
    best_gate_history = None
    best_model = None

    for config in configs:
        model, history, test_acc, sparsity_report, gate_history = train_prunable(
            config, train_loader, test_loader, device, epochs, output_dir
        )

        # Compute additional metrics
        flops_reduction = compute_flops_reduction(model.to(device), baseline_model.to(device))
        latency = measure_latency(model.to(device), device, n_runs=50)
        baseline_latency = measure_latency(baseline_model.to(device), device, n_runs=50)

        result = {
            "name": config["name"],
            "lambda_max": config["lambda_max"],
            "schedule_mode": config.get("schedule_mode", "static"),
            "test_accuracy": test_acc,
            "baseline_accuracy": baseline_acc,
            "sparsity": sparsity_report.get("overall", 0.0),
            "sparsity_report": sparsity_report,
            "flops_reduction_pct": flops_reduction,
            "latency_ms": latency,
            "baseline_latency_ms": baseline_latency,
        }
        all_results.append(result)
        all_histories[config["name"]] = history

        # Track best for visualization
        if best_gate_history is None or len(gate_history) > len(best_gate_history):
            best_gate_history = gate_history
            best_model = model

    # Also save the overall best model as best_model.pth for API
    if all_results:
        best_result = max(all_results, key=lambda r: r["test_accuracy"])
        best_path = output_dir / f"{best_result['name']}_best.pth"
        if best_path.exists():
            import shutil
            shutil.copy2(best_path, output_dir / "best_model.pth")

    # ---- Save results ----
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "baseline_accuracy": baseline_acc,
                "experiments": all_results,
                "epochs": epochs,
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved: {results_path}")

    # ---- Generate visualizations ----
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    if best_model is not None:
        gate_values = best_model.get_layer_gate_values()

        # Figure 1: Gate distribution
        plot_gate_distribution(
            gate_values,
            save_path=str(output_dir / "gate_distribution.png"),
        )

        # Figure 2: Layer heatmaps
        plot_layer_heatmaps(
            gate_values,
            save_path=str(output_dir / "layer_heatmaps.png"),
        )

    # Figure 3: Lambda trade-off
    if all_results:
        plot_lambda_tradeoff(
            all_results,
            save_path=str(output_dir / "lambda_tradeoff.png"),
        )

    # Figure 4: Training curves
    if all_histories:
        plot_training_curves(
            all_histories,
            save_path=str(output_dir / "training_curves.png"),
        )

    # Animated GIF
    if best_gate_history:
        animate_gate_evolution(
            best_gate_history,
            save_path=str(output_dir / "gate_evolution.gif"),
        )

    # ---- Print summary table ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline accuracy: {baseline_acc:.4f}")
    print(f"\n  {'Config':<20} {'Acc':>8} {'Sparsity':>10} {'FLOPs↓':>8} {'Latency':>10}")
    print("  " + "-" * 60)
    for r in all_results:
        print(
            f"  {r['name']:<20} "
            f"{r['test_accuracy']:>7.4f} "
            f"{r['sparsity']:>9.1%} "
            f"{r['flops_reduction_pct']:>7.1f}% "
            f"{r['latency_ms']:>8.2f}ms"
        )

    print(f"\n  All outputs saved to: {output_dir}")
    print("  Done!")


if __name__ == "__main__":
    main()
