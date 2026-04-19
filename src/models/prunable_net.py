"""
PrunableNet — 4-layer deep feedforward network with skip connections
for CIFAR-10 classification using PrunableLinear layers.

Architecture:
    Input (3072) → PrunableLinear(1024) → BN → GELU
                 → PrunableLinear(512)  → BN → GELU
                 → PrunableLinear(256)  → BN → GELU + skip(3072→256)
                 → PrunableLinear(128)  → GELU
                 → Linear(10) [classifier head, NOT prunable]

The skip connection from input to layer 3 output makes pruning more
interesting: the network can learn to prune entire middle pathways
while maintaining accuracy through the residual path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.prunable_linear import PrunableLinear


class PrunableNet(nn.Module):
    """Self-pruning feedforward network for CIFAR-10.

    Args:
        gate_strategy: Gate activation for all PrunableLinear layers.
            One of 'sigmoid', 'hard_sigmoid', 'ste'.
        temperature: Temperature for gate sharpness.
    """

    def __init__(self, gate_strategy: str = "sigmoid", temperature: float = 1.0):
        super().__init__()
        self.flatten = nn.Flatten()

        # Prunable layers — CIFAR-10 images: 32×32×3 = 3072 input features
        self.layer1 = PrunableLinear(3072, 1024, gate_strategy, temperature)
        self.layer2 = PrunableLinear(1024, 512, gate_strategy, temperature)
        self.layer3 = PrunableLinear(512, 256, gate_strategy, temperature)
        self.layer4 = PrunableLinear(256, 128, gate_strategy, temperature)

        # Classifier head — standard Linear, NOT prunable
        self.head = nn.Linear(128, 10)

        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        # Skip connection: projects input directly to layer3 output dim
        # This creates an alternative path that makes pruning more interesting
        self.skip_proj = nn.Linear(3072, 256, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x: Input images of shape (batch, 3, 32, 32).

        Returns:
            Logits of shape (batch, 10).
        """
        x = self.flatten(x)  # (batch, 3072)

        # Save for skip connection
        skip = self.skip_proj(x)  # (batch, 256)

        # Main pathway with prunable layers
        x = F.gelu(self.bn1(self.layer1(x)))  # (batch, 1024)
        x = F.gelu(self.bn2(self.layer2(x)))  # (batch, 512)
        x = F.gelu(self.bn3(self.layer3(x)))  # (batch, 256)

        # Residual addition — if layers 1-3 are heavily pruned,
        # the skip connection maintains information flow
        x = x + skip

        x = F.gelu(self.layer4(x))  # (batch, 128)
        return self.head(x)  # (batch, 10)

    def get_all_gates(self) -> torch.Tensor:
        """Returns concatenated gate values for all prunable layers.

        Used by SparsityLoss to compute the regularization penalty.

        Returns:
            1D tensor of all gate values across all prunable layers.
        """
        gate_values = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gate_values.append(module.compute_gates().flatten())
        return torch.cat(gate_values)

    def get_sparsity_report(self, threshold: float = 1e-2) -> dict:
        """Get per-layer and overall sparsity statistics.

        Args:
            threshold: Gate value below which a weight is "pruned".

        Returns:
            Dict with per-layer sparsity and overall average.
        """
        report = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                report[name] = module.get_sparsity(threshold)
        if report:
            report["overall"] = sum(report.values()) / len(report)
        return report

    def get_layer_gate_values(self, threshold: float = 1e-2) -> dict:
        """Get gate values for each prunable layer (for visualization).

        Returns:
            Dict mapping layer names to gate value tensors.
        """
        gate_values = {}
        for name, module in self.named_modules():
            if isinstance(module, PrunableLinear):
                with torch.no_grad():
                    gate_values[name] = module.compute_gates().cpu()
        return gate_values

    def get_total_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_active_params(self, threshold: float = 1e-2) -> int:
        """Number of active (non-pruned) weight parameters."""
        active = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                active += module.get_active_params(threshold)
            elif isinstance(module, nn.Linear):
                active += module.weight.numel()
                if module.bias is not None:
                    active += module.bias.numel()
        return active
