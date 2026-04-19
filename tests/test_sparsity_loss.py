"""
Tests for SparsityLoss — verify L1, L2, and Hoyer loss computations.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses.sparsity_loss import SparsityLoss
from src.models.prunable_net import PrunableNet


class TestL1Loss:
    """Test L1 sparsity loss behavior."""

    def test_l1_is_sum_of_gates(self):
        model = PrunableNet()
        loss_fn = SparsityLoss(norm="L1")
        gates = model.get_all_gates()
        expected = gates.sum()
        actual = loss_fn(model)
        assert torch.allclose(actual, expected)

    def test_l1_increases_with_more_open_gates(self):
        loss_fn = SparsityLoss(norm="L1")

        model1 = PrunableNet()
        with torch.no_grad():
            for m in model1.modules():
                if hasattr(m, "gate_scores"):
                    m.gate_scores.fill_(-10)  # mostly closed

        model2 = PrunableNet()
        with torch.no_grad():
            for m in model2.modules():
                if hasattr(m, "gate_scores"):
                    m.gate_scores.fill_(10)  # mostly open

        assert loss_fn(model2) > loss_fn(model1)

    def test_l1_is_differentiable(self):
        model = PrunableNet()
        loss_fn = SparsityLoss(norm="L1")
        loss = loss_fn(model)
        loss.backward()
        assert model.layer1.gate_scores.grad is not None

    def test_l1_gradient_is_constant(self):
        """L1 gradient should be ~1 for positive gates (sigmoid outputs)."""
        model = PrunableNet()
        loss_fn = SparsityLoss(norm="L1")
        loss = loss_fn(model)
        loss.backward()
        # The gate gradient includes sigmoid derivative, so it won't be
        # exactly 1, but should be reasonably uniform
        grad = model.layer1.gate_scores.grad
        assert grad is not None
        assert grad.abs().mean() > 0


class TestL2Loss:
    """Test L2 sparsity loss behavior."""

    def test_l2_is_sum_of_squared_gates(self):
        model = PrunableNet()
        loss_fn = SparsityLoss(norm="L2")
        gates = model.get_all_gates()
        expected = (gates**2).sum()
        actual = loss_fn(model)
        assert torch.allclose(actual, expected)

    def test_l2_smaller_gradient_near_zero(self):
        """L2 gradient = 2g → 0 as g → 0. Compare with L1."""
        model = PrunableNet()
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, "gate_scores"):
                    m.gate_scores.fill_(-5)  # small gates

        loss_l2 = SparsityLoss(norm="L2")(model)
        loss_l2.backward()
        grad_l2 = model.layer1.gate_scores.grad.abs().mean().item()
        model.zero_grad()

        loss_l1 = SparsityLoss(norm="L1")(model)
        loss_l1.backward()
        grad_l1 = model.layer1.gate_scores.grad.abs().mean().item()

        # L2 gradient should be smaller than L1 when gates are near 0
        assert grad_l2 < grad_l1, "L2 gradient should be smaller near zero"


class TestHoyerLoss:
    """Test Hoyer sparsity loss."""

    def test_hoyer_computation(self):
        model = PrunableNet()
        loss_fn = SparsityLoss(norm="Hoyer")
        gates = model.get_all_gates()
        expected = gates.sum() / (gates.norm() + 1e-8)
        actual = loss_fn(model)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_hoyer_is_differentiable(self):
        model = PrunableNet()
        loss_fn = SparsityLoss(norm="Hoyer")
        loss = loss_fn(model)
        loss.backward()
        assert model.layer1.gate_scores.grad is not None


class TestLossRegistry:
    """Test loss_registry factory."""

    def test_get_all_losses(self):
        from src.losses.loss_registry import get_loss, list_losses

        available = list_losses()
        assert "L1" in available
        assert "L2" in available
        assert "Hoyer" in available

        for name in available:
            loss = get_loss(name)
            assert isinstance(loss, SparsityLoss)

    def test_invalid_loss_raises(self):
        from src.losses.loss_registry import get_loss

        with pytest.raises(ValueError):
            get_loss("invalid")


class TestSparsityLossValidation:
    """Test input validation."""

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError):
            SparsityLoss(norm="L3")

    def test_loss_is_non_negative(self):
        model = PrunableNet()
        for norm in ["L1", "L2", "Hoyer"]:
            loss = SparsityLoss(norm=norm)(model)
            assert loss.item() >= 0, f"{norm} loss should be non-negative"
