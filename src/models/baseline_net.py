"""
BaselineNet — Identical architecture to PrunableNet but using standard
nn.Linear layers (no gates). Used as a comparison baseline for
evaluation metrics (accuracy, FLOPs, latency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineNet(nn.Module):
    """Standard feedforward network for CIFAR-10 (no pruning).

    Mirrors PrunableNet architecture exactly for fair comparison:
    3072 → 1024 → 512 → 256 → 128 → 10 with skip connection.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.layer1 = nn.Linear(3072, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        self.head = nn.Linear(128, 10)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        self.skip_proj = nn.Linear(3072, 256, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        skip = self.skip_proj(x)

        x = F.gelu(self.bn1(self.layer1(x)))
        x = F.gelu(self.bn2(self.layer2(x)))
        x = F.gelu(self.bn3(self.layer3(x)))
        x = x + skip
        x = F.gelu(self.layer4(x))
        return self.head(x)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
