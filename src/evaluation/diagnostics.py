"""
Gradient flow diagnostics — verify that gradients reach gate_scores.

This is critical for the self-pruning mechanism: if gradients don't
flow to gate_scores, the gates can't learn which weights to prune.
"""

import torch.nn as nn

from src.layers.prunable_linear import PrunableLinear


def verify_gradient_flow(model: nn.Module) -> dict:
    """Register gradient hooks on all PrunableLinear gate_scores and weights.

    Usage:
        grad_norms = verify_gradient_flow(model)
        loss = model(x).sum()
        loss.backward()
        # Now grad_norms is populated
        print(grad_norms)

    Args:
        model: Model with PrunableLinear layers.

    Returns:
        Dict that will be populated with gradient norms after backward().
        Keys are like "layer1.gate_scores", "layer1.weight".
    """
    grad_norms = {}

    def make_hook(name: str):
        def hook(grad):
            grad_norms[name] = grad.norm().item()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, PrunableLinear):
            module.gate_scores.register_hook(make_hook(f"{name}.gate_scores"))
            module.weight.register_hook(make_hook(f"{name}.weight"))

    return grad_norms


def get_gate_statistics(model: nn.Module) -> dict:
    """Get detailed statistics about gate values in each layer.

    Args:
        model: Model with PrunableLinear layers.

    Returns:
        Dict with per-layer gate stats (mean, std, min, max, % near 0, % near 1).
    """
    import torch

    stats = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, PrunableLinear):
                gates = module.compute_gates()
                stats[name] = {
                    "mean": gates.mean().item(),
                    "std": gates.std().item(),
                    "min": gates.min().item(),
                    "max": gates.max().item(),
                    "pct_near_zero": (gates < 0.01).float().mean().item(),
                    "pct_near_one": (gates > 0.99).float().mean().item(),
                    "shape": tuple(gates.shape),
                }
    return stats
