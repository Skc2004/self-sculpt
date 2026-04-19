"""
Tests for end-to-end gradient flow through the full PrunableNet.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.prunable_net import PrunableNet
from src.layers.prunable_linear import PrunableLinear
from src.evaluation.diagnostics import verify_gradient_flow


class TestFullNetworkGradientFlow:
    """Verify gradients reach gate_scores in every layer of PrunableNet."""

    def test_all_layers_receive_gradients(self):
        model = PrunableNet()
        grad_norms = verify_gradient_flow(model)

        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Check that gate_scores got gradients in every prunable layer
        gate_keys = [k for k in grad_norms if "gate_scores" in k]
        assert len(gate_keys) == 4, f"Expected 4 gate layers, got {len(gate_keys)}"

        for key in gate_keys:
            assert grad_norms[key] > 0, f"Zero gradient at {key}"

    def test_weight_gradients_exist(self):
        model = PrunableNet()
        grad_norms = verify_gradient_flow(model)

        x = torch.randn(4, 3, 32, 32)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        weight_keys = [k for k in grad_norms if "weight" in k]
        for key in weight_keys:
            assert grad_norms[key] > 0, f"Zero gradient at {key}"

    def test_gradient_magnitude_reasonable(self):
        """Gradients should not be exploding or vanishing badly."""
        model = PrunableNet()
        grad_norms = verify_gradient_flow(model)

        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        for key, norm in grad_norms.items():
            assert norm < 1e6, f"Exploding gradient at {key}: {norm}"
            assert norm > 1e-10, f"Vanishing gradient at {key}: {norm}"

    def test_sparsity_loss_adds_gradient_to_gates(self):
        """Adding sparsity loss should increase gradient magnitude on gate_scores."""
        from src.losses.sparsity_loss import SparsityLoss

        model = PrunableNet()
        sparsity_loss = SparsityLoss(norm="L1")

        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))

        # Without sparsity loss
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad_without = model.layer1.gate_scores.grad.norm().item()
        model.zero_grad()

        # With sparsity loss
        logits = model(x)
        loss = F.cross_entropy(logits, y) + 0.01 * sparsity_loss(model)
        loss.backward()
        grad_with = model.layer1.gate_scores.grad.norm().item()

        assert grad_with > grad_without, (
            f"Sparsity loss should increase gate gradient: "
            f"{grad_with} <= {grad_without}"
        )


class TestForwardPassCorrectness:
    """Test that PrunableNet forward pass is correct."""

    def test_output_shape(self):
        model = PrunableNet()
        x = torch.randn(8, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (8, 10)

    def test_skip_connection_works(self):
        """With main path heavily pruned, skip connection should maintain output."""
        model = PrunableNet()
        # Close gates on layers 1-3
        with torch.no_grad():
            for layer in [model.layer1, model.layer2, model.layer3]:
                layer.gate_scores.fill_(-100)

        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        # Output should still have reasonable values (not all zero)
        assert out.abs().mean() > 1e-4, "Output is nearly zero even with skip connection"

    def test_sparsity_report_structure(self):
        model = PrunableNet()
        report = model.get_sparsity_report()
        assert "overall" in report
        assert len(report) == 5  # 4 layers + overall

    def test_get_all_gates(self):
        model = PrunableNet()
        gates = model.get_all_gates()
        # Total gates = 3072*1024 + 1024*512 + 512*256 + 256*128
        expected = 3072 * 1024 + 1024 * 512 + 512 * 256 + 256 * 128
        assert gates.shape == (expected,)
