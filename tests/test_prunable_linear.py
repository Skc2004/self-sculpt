"""
Tests for PrunableLinear layer — the core custom module.
"""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layers.prunable_linear import PrunableLinear


class TestPrunableLinearShape:
    """Test that PrunableLinear produces correct output shapes."""

    def test_output_shape_basic(self):
        layer = PrunableLinear(64, 32)
        x = torch.randn(8, 64)
        assert layer(x).shape == (8, 32)

    def test_output_shape_single_sample(self):
        layer = PrunableLinear(128, 10)
        x = torch.randn(1, 128)
        assert layer(x).shape == (1, 10)

    def test_output_shape_large_batch(self):
        layer = PrunableLinear(3072, 1024)
        x = torch.randn(64, 3072)
        assert layer(x).shape == (64, 1024)


class TestGateProperties:
    """Test that gates have correct mathematical properties."""

    def test_gates_in_range_sigmoid(self):
        layer = PrunableLinear(64, 32, gate_strategy="sigmoid")
        gates = layer.compute_gates()
        assert gates.min() >= 0.0, f"Gate min {gates.min()} < 0"
        assert gates.max() <= 1.0, f"Gate max {gates.max()} > 1"

    def test_gates_in_range_hard_sigmoid(self):
        layer = PrunableLinear(64, 32, gate_strategy="hard_sigmoid")
        gates = layer.compute_gates()
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0

    def test_gates_in_range_ste(self):
        layer = PrunableLinear(64, 32, gate_strategy="ste")
        gates = layer.compute_gates()
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0

    def test_initial_gates_mostly_open(self):
        """Gate scores initialized to 0.5, so sigmoid(0.5) ≈ 0.62."""
        layer = PrunableLinear(64, 32, gate_strategy="sigmoid")
        gates = layer.compute_gates()
        mean_gate = gates.mean().item()
        assert 0.55 < mean_gate < 0.70, f"Initial mean gate {mean_gate} not near 0.62"

    def test_gate_shape_matches_weight(self):
        layer = PrunableLinear(64, 32)
        assert layer.gate_scores.shape == layer.weight.shape


class TestGradientFlow:
    """Test that gradients flow correctly through gate_scores."""

    def test_gradient_flows_to_gate_scores(self):
        layer = PrunableLinear(64, 32)
        x = torch.randn(4, 64)
        loss = layer(x).sum()
        loss.backward()
        assert layer.gate_scores.grad is not None, "No gradient on gate_scores"
        assert layer.gate_scores.grad.abs().sum().item() > 0, "Zero gradient on gate_scores"

    def test_gradient_flows_to_weights(self):
        layer = PrunableLinear(64, 32)
        x = torch.randn(4, 64)
        loss = layer(x).sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum().item() > 0

    def test_gradient_flows_all_strategies(self):
        for strategy in ["sigmoid", "hard_sigmoid", "ste"]:
            layer = PrunableLinear(64, 32, gate_strategy=strategy)
            # Ensure gates are not saturated for hard_sigmoid gradient test
            with torch.no_grad():
                layer.gate_scores.fill_(0.1)
            x = torch.randn(4, 64)
            loss = layer(x).sum()
            loss.backward()
            assert layer.gate_scores.grad is not None, f"No grad for {strategy}"
            assert layer.gate_scores.grad.abs().sum().item() > 0, f"Zero grad for {strategy}"


class TestPruningBehavior:
    """Test that pruning via gates works as expected."""

    def test_fully_closed_gates_zero_output(self):
        """With gates ≈ 0, only bias contributes to the output."""
        layer = PrunableLinear(64, 32)
        with torch.no_grad():
            layer.gate_scores.fill_(-100.0)  # sigmoid(-100) ≈ 0
        x = torch.randn(4, 64)
        out = layer(x)
        # Output should be approximately equal to just the bias
        bias_expanded = layer.bias.unsqueeze(0).expand_as(out)
        assert (out - bias_expanded).abs().max() < 1e-4

    def test_fully_open_gates_match_linear(self):
        """With gates ≈ 1, should behave like a standard linear layer."""
        layer = PrunableLinear(64, 32)
        with torch.no_grad():
            layer.gate_scores.fill_(100.0)  # sigmoid(100) ≈ 1
        x = torch.randn(4, 64)
        out_prunable = layer(x)
        out_linear = torch.nn.functional.linear(x, layer.weight, layer.bias)
        assert (out_prunable - out_linear).abs().max() < 1e-4

    def test_sparsity_increases_with_closed_gates(self):
        layer = PrunableLinear(64, 32)
        # Start: gates are open
        sparsity_open = layer.get_sparsity()

        # Close half the gates
        with torch.no_grad():
            layer.gate_scores[:16, :] = -100.0
        sparsity_half = layer.get_sparsity()

        assert sparsity_half > sparsity_open

    def test_sparsity_report_value(self):
        layer = PrunableLinear(64, 32)
        with torch.no_grad():
            layer.gate_scores.fill_(-100.0)
        sparsity = layer.get_sparsity()
        assert sparsity > 0.99  # Almost all gates closed


class TestTemperature:
    """Test temperature scaling effects on gates."""

    def test_low_temperature_sharper_gates(self):
        """Lower temperature should make gates more binary."""
        layer_warm = PrunableLinear(64, 32, temperature=10.0)
        layer_cold = PrunableLinear(64, 32, temperature=0.1)

        # Randomize gate_scores to measure standard deviation across values
        with torch.no_grad():
            random_scores = torch.randn(32, 64)
            layer_warm.gate_scores.copy_(random_scores)
            layer_cold.gate_scores.copy_(random_scores)

        # Both initialized with same gate_scores
        gates_warm = layer_warm.compute_gates()
        gates_cold = layer_cold.compute_gates()

        # Cold gates should be more spread (closer to 0 or 1)
        std_warm = gates_warm.std().item()
        std_cold = gates_cold.std().item()
        assert std_cold > std_warm, "Low temperature should produce sharper gates"


class TestNoBias:
    """Test PrunableLinear without bias."""

    def test_no_bias(self):
        layer = PrunableLinear(64, 32, bias=False)
        assert layer.bias is None
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_no_bias_closed_gates(self):
        """Without bias and with gates closed, output should be zero."""
        layer = PrunableLinear(64, 32, bias=False)
        with torch.no_grad():
            layer.gate_scores.fill_(-100.0)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.abs().max() < 1e-4


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown gate strategy"):
            PrunableLinear(64, 32, gate_strategy="invalid")

    def test_single_neuron(self):
        layer = PrunableLinear(1, 1)
        x = torch.randn(1, 1)
        out = layer(x)
        assert out.shape == (1, 1)

    def test_extra_repr(self):
        layer = PrunableLinear(64, 32, gate_strategy="ste", temperature=0.5)
        repr_str = layer.extra_repr()
        assert "64" in repr_str
        assert "32" in repr_str
        assert "ste" in repr_str
