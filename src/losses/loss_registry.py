"""
Loss registry — Factory pattern for selecting sparsity loss by name.
"""

from .sparsity_loss import SparsityLoss


_REGISTRY = {
    "L1": lambda: SparsityLoss(norm="L1"),
    "L2": lambda: SparsityLoss(norm="L2"),
    "Hoyer": lambda: SparsityLoss(norm="Hoyer"),
}


def get_loss(name: str) -> SparsityLoss:
    """Create a sparsity loss by name.

    Args:
        name: One of 'L1', 'L2', 'Hoyer'.

    Returns:
        SparsityLoss instance.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def list_losses() -> list[str]:
    """Return available loss names."""
    return list(_REGISTRY.keys())
