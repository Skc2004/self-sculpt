"""
Evaluation metrics — accuracy, FLOPs, latency, and full reporting.
"""

import time

import torch
import torch.nn as nn

from src.layers.prunable_linear import PrunableLinear


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader, device: str = "cpu") -> float:
    """Compute classification accuracy on a dataset.

    Args:
        model: Trained model.
        loader: DataLoader for evaluation.
        device: Torch device.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def count_flops(model: nn.Module, threshold: float = 1e-2) -> int:
    """Count active FLOPs after pruning (skip zeroed-out weights).

    For each PrunableLinear layer, only count weights where
    gate >= threshold. Each active weight contributes 2 FLOPs
    (one multiply + one add).

    For standard nn.Linear layers, count all weights.

    Args:
        model: Model to analyze.
        threshold: Gate value below which a weight is "pruned".

    Returns:
        Total active FLOPs.
    """
    flops = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.compute_gates()
            active_weights = (gates >= threshold).float().sum().item()
            flops += int(2 * active_weights)  # multiply + add
        elif isinstance(module, nn.Linear):
            flops += 2 * module.weight.numel()
    return flops


@torch.no_grad()
def count_active_params(model: nn.Module, threshold: float = 1e-2) -> int:
    """Count active (non-pruned) parameters.

    Args:
        model: Model to analyze.
        threshold: Gate threshold for PrunableLinear layers.

    Returns:
        Number of active parameters.
    """
    active = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = module.compute_gates()
            active += int((gates >= threshold).sum().item())
            if module.bias is not None:
                active += module.bias.numel()
        elif isinstance(module, nn.Linear):
            active += module.weight.numel()
            if module.bias is not None:
                active += module.bias.numel()
        elif isinstance(module, nn.BatchNorm1d):
            if module.weight is not None:
                active += module.weight.numel()
            if module.bias is not None:
                active += module.bias.numel()
    return active


def compute_flops_reduction(
    model: nn.Module, baseline: nn.Module, threshold: float = 1e-2
) -> float:
    """Compute FLOPs reduction percentage vs baseline.

    Args:
        model: Pruned model.
        baseline: Unpruned baseline model.
        threshold: Gate threshold.

    Returns:
        Reduction percentage (e.g., 45.2 means 45.2% fewer FLOPs).
    """
    pruned_flops = count_flops(model, threshold)
    baseline_flops = count_flops(baseline)
    if baseline_flops == 0:
        return 0.0
    return (1 - pruned_flops / baseline_flops) * 100


def measure_latency(
    model: nn.Module, device: str = "cpu", n_runs: int = 100, batch_size: int = 32
) -> float:
    """Measure mean forward pass latency in milliseconds.

    Includes warmup runs to account for JIT compilation and caching.

    Args:
        model: Model to benchmark.
        device: Torch device.
        n_runs: Number of timed forward passes.
        batch_size: Batch size for benchmarking.

    Returns:
        Mean latency in milliseconds.
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, 32, 32).to(device)

    # Warmup
    for _ in range(10):
        model(dummy)

    if device == "cuda" and torch.cuda.is_available():
        # CUDA timing with events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_runs):
            model(dummy)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_runs
    else:
        # CPU timing
        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms / n_runs


def full_evaluation_report(
    model: nn.Module,
    test_loader,
    baseline_model: nn.Module,
    device: str = "cpu",
    threshold: float = 1e-2,
) -> dict:
    """Generate a comprehensive evaluation report.

    Args:
        model: Trained pruned model.
        test_loader: Test DataLoader.
        baseline_model: Trained baseline model.
        device: Torch device.
        threshold: Gate threshold for sparsity metrics.

    Returns:
        Dict with all evaluation metrics.
    """
    model.to(device).eval()
    baseline_model.to(device).eval()

    sparsity_report = {}
    if hasattr(model, "get_sparsity_report"):
        sparsity_report = model.get_sparsity_report(threshold)

    total_params = sum(p.numel() for p in model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())

    return {
        "test_accuracy": evaluate_accuracy(model, test_loader, device),
        "baseline_accuracy": evaluate_accuracy(baseline_model, test_loader, device),
        "sparsity_report": sparsity_report,
        "total_params": total_params,
        "baseline_params": baseline_params,
        "active_params": count_active_params(model, threshold),
        "flops_pruned": count_flops(model, threshold),
        "flops_baseline": count_flops(baseline_model),
        "flops_reduction_pct": compute_flops_reduction(model, baseline_model, threshold),
        "latency_ms": measure_latency(model, device),
        "baseline_latency_ms": measure_latency(baseline_model, device),
    }
