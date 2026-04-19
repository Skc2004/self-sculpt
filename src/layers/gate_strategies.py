"""
Gate strategy functions for PrunableLinear layers.

Each strategy maps raw gate_scores → [0, 1] gate values with different
gradient properties:

- **sigmoid**: Smooth, differentiable everywhere. Standard choice.
  Gradient: sigmoid'(s) = sigmoid(s)(1 - sigmoid(s))

- **hard_sigmoid**: Piecewise linear approximation. Faster computation,
  but gradients are zero outside [-0.5/T, 0.5/T] range.

- **ste** (Straight-Through Estimator): Binary gates in the forward pass
  (hard 0/1), but uses sigmoid gradient in the backward pass.
  Enables true discrete sparsity during inference while maintaining
  gradient flow for training.
"""

import torch
import torch.nn.functional as F


def sigmoid_gate(gate_scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Standard sigmoid gating with temperature scaling.

    Args:
        gate_scores: Raw learnable parameters.
        temperature: Controls sharpness. Lower T → sharper (more binary).

    Returns:
        Gate values in (0, 1).
    """
    return torch.sigmoid(gate_scores / temperature)


def hard_sigmoid_gate(gate_scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Hard sigmoid: piecewise linear approximation clamped to [0, 1].

    Faster than sigmoid and produces exact 0/1 at extremes.

    Args:
        gate_scores: Raw learnable parameters.
        temperature: Controls the slope of the linear region.

    Returns:
        Gate values in [0, 1].
    """
    return F.hardtanh(gate_scores / temperature + 0.5, 0.0, 1.0)


def ste_gate(gate_scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Straight-Through Estimator: binary forward, smooth backward.

    Forward pass: hard threshold → {0, 1}
    Backward pass: gradient flows through sigmoid(gate_scores)

    This is the STE trick from Bengio et al. (2013).

    Args:
        gate_scores: Raw learnable parameters.
        temperature: Controls sigmoid sharpness in backward pass.

    Returns:
        Binary gate values {0, 1} with smooth gradients.
    """
    soft = torch.sigmoid(gate_scores / temperature)
    hard = (gate_scores > 0).float()
    # Detach the difference so gradients flow through `soft` only
    return soft + (hard - soft).detach()


# Registry for lookup by name
GATE_STRATEGIES = {
    "sigmoid": sigmoid_gate,
    "hard_sigmoid": hard_sigmoid_gate,
    "ste": ste_gate,
}


def get_gate_fn(name: str):
    """Get a gate strategy function by name.

    Args:
        name: One of 'sigmoid', 'hard_sigmoid', 'ste'.

    Returns:
        Callable(gate_scores, temperature) -> gate_values
    """
    if name not in GATE_STRATEGIES:
        raise ValueError(
            f"Unknown gate strategy '{name}'. "
            f"Available: {list(GATE_STRATEGIES.keys())}"
        )
    return GATE_STRATEGIES[name]
