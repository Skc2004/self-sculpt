"""
Lambda Scheduler — Controls the sparsity penalty strength (λ) over training.

Key insight: start with λ=0 during a warmup period so the network can learn
useful representations first, then gradually increase the pruning pressure.
This prevents premature pruning of important connections.

Modes:
- static:  constant λ after warmup
- linear:  linearly ramp from 0 to λ_max after warmup
- cosine:  cosine annealing from 0 to λ_max after warmup (slower start, faster end)
"""

import math


class LambdaScheduler:
    """Schedules the sparsity regularization coefficient λ over epochs.

    Args:
        lambda_max: Maximum λ value to reach.
        warmup_epochs: Number of initial epochs with λ=0.
        total_epochs: Total training epochs.
        mode: Scheduling mode — 'static', 'linear', or 'cosine'.
    """

    VALID_MODES = ("static", "linear", "cosine")

    def __init__(
        self,
        lambda_max: float,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        mode: str = "linear",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.mode = mode

    def get_lambda(self, epoch: int) -> float:
        """Get the λ value for the current epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Lambda value for this epoch.
        """
        # During warmup: let the network learn, no pruning
        if epoch < self.warmup_epochs:
            return 0.0

        # After warmup: apply scheduling
        if self.mode == "static":
            return self.lambda_max

        # Progress from 0 to 1 over post-warmup period
        remaining = self.total_epochs - self.warmup_epochs
        if remaining <= 0:
            return self.lambda_max
        progress = (epoch - self.warmup_epochs) / remaining
        progress = min(progress, 1.0)

        if self.mode == "linear":
            return self.lambda_max * progress
        elif self.mode == "cosine":
            # Cosine curve: slow start, fast finish
            return self.lambda_max * (1 - math.cos(math.pi * progress)) / 2

        return self.lambda_max

    def __repr__(self) -> str:
        return (
            f"LambdaScheduler(λ_max={self.lambda_max}, "
            f"warmup={self.warmup_epochs}, "
            f"total={self.total_epochs}, "
            f"mode={self.mode})"
        )
