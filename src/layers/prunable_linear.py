"""
PrunableLinear — A fully connected layer with learnable, differentiable
gate parameters that enable automatic weight pruning during training.

Each weight w_ij is multiplied by a gate g_ij ∈ [0, 1]:
    output = x @ (W ⊙ G)^T + b

The gates are derived from learnable `gate_scores` via a differentiable
activation (sigmoid by default). During training, an L1 penalty on the
gate values pushes them toward 0, effectively pruning the corresponding
weights. The gradient update for gate_scores is:

    s ← s - lr × [∂L_cls/∂s + λ × ∂L_sparse/∂s]

where ∂L_sparse/∂s = sigmoid'(s) for L1 on sigmoid gates.

Design choice: gate_scores are initialized to 0.5 so that sigmoid(0.5) ≈ 0.62,
meaning most gates start open. This lets the network learn representations
before the sparsity penalty kicks in (especially with warmup scheduling).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gate_strategies import get_gate_fn


class PrunableLinear(nn.Module):
    """Linear layer with per-weight differentiable gate parameters.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        gate_strategy: Gate activation function. One of
            'sigmoid', 'hard_sigmoid', 'ste'.
        temperature: Temperature parameter for gating sharpness.
            Lower values produce sharper (more binary) gates.
        bias: If True, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        gate_strategy: str = "sigmoid",
        temperature: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.gate_strategy = gate_strategy

        # Core parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Gate function
        self._gate_fn = get_gate_fn(gate_strategy)

        # Initialize
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights and gate scores.

        Weights use Kaiming uniform (standard for ReLU/GELU networks).
        Gate scores are set to 0.5 so sigmoid(0.5) ≈ 0.62 — most gates
        start open, allowing the network to learn before pruning.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Gates start mostly open
        nn.init.constant_(self.gate_scores, 0.5)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def compute_gates(self) -> torch.Tensor:
        """Compute gate values from raw gate_scores.

        Returns:
            Tensor of shape (out_features, in_features) with values in [0, 1].
        """
        return self._gate_fn(self.gate_scores, self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated weights.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        gates = self.compute_gates()  # (out, in)
        pruned_weights = self.weight * gates  # element-wise gating
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """Compute the fraction of weights that are effectively pruned.

        A weight is considered pruned if its gate value < threshold.

        Args:
            threshold: Gate value below which a weight is "pruned".

        Returns:
            Sparsity ratio in [0, 1]. Higher = more pruned.
        """
        with torch.no_grad():
            gates = self.compute_gates()
            return (gates < threshold).float().mean().item()

    def get_active_params(self, threshold: float = 1e-2) -> int:
        """Count the number of active (non-pruned) parameters.

        Args:
            threshold: Gate value below which a weight is "pruned".

        Returns:
            Number of active weight parameters.
        """
        with torch.no_grad():
            gates = self.compute_gates()
            return int((gates >= threshold).sum().item())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"gate_strategy={self.gate_strategy}, "
            f"temperature={self.temperature}, "
            f"bias={self.bias is not None}"
        )
