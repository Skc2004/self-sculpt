"""
Sparsity Loss — Regularization penalties on gate values to encourage pruning.

Mathematical justification for L1 vs L2:

**L1 (sum of absolute values):**
    L_sparse = Σ |g_i|
    ∂L/∂g = sign(g) = 1   (for g > 0, which sigmoid outputs always are)

    The gradient is CONSTANT regardless of how small g gets.
    This means gates are pushed toward 0 with equal force at all magnitudes,
    so they actually reach (near) zero → TRUE sparsity.

**L2 (sum of squared values):**
    L_sparse = Σ g_i²
    ∂L/∂g = 2g → 0 as g → 0

    The gradient DIMINISHES as g approaches 0. Gates get small but
    never truly reach zero → soft sparsity, not hard pruning.

**Hoyer (L1/L2 ratio):**
    L_sparse = (Σ |g_i|) / (√Σ g_i² + ε)

    Encourages structured sparsity — prefers many exact zeros
    over many small values. The ratio is minimized when all gates
    are either 0 or 1 (binary).
"""

import torch
import torch.nn as nn


class SparsityLoss(nn.Module):
    """Differentiable sparsity penalty on gate values.

    Args:
        norm: Type of sparsity norm. One of 'L1', 'L2', 'Hoyer'.
    """

    VALID_NORMS = ("L1", "L2", "Hoyer")

    def __init__(self, norm: str = "L1"):
        super().__init__()
        if norm not in self.VALID_NORMS:
            raise ValueError(f"norm must be one of {self.VALID_NORMS}, got '{norm}'")
        self.norm = norm

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute sparsity loss from model's gate values.

        Args:
            model: A model with a `get_all_gates()` method (e.g., PrunableNet).

        Returns:
            Scalar loss tensor.
        """
        gates = model.get_all_gates()

        if self.norm == "L1":
            return gates.sum()
        elif self.norm == "L2":
            return (gates**2).sum()
        elif self.norm == "Hoyer":
            # Hoyer sparsity measure — ratio of L1 to L2 norm
            # Lower when distribution is more sparse (more exact zeros)
            l1 = gates.sum()
            l2 = gates.norm() + 1e-8  # avoid division by zero
            return l1 / l2

    def extra_repr(self) -> str:
        return f"norm={self.norm}"
