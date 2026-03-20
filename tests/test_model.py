"""Unit tests for MNISTNet (CNN) architecture."""

import torch
import pytest

from classifiers.datasets.mnist.models import MNISTNet
from classifiers.base_model import BaseModel


class TestMNISTNet:
    def test_output_shape_single(self, untrained_model):
        x = torch.randn(1, 1, 28, 28)
        out = untrained_model(x)
        assert out.shape == (1, 10)

    def test_output_shape_batch(self, untrained_model, sample_batch):
        out = untrained_model(sample_batch)
        assert out.shape == (4, 10)

    def test_output_is_logits_not_probabilities(self, untrained_model):
        """Output should be raw logits (can be negative, don't sum to 1)."""
        x = torch.randn(1, 1, 28, 28)
        out = untrained_model(x)
        assert out.min().item() < 1.0 or out.max().item() > 0.0

    def test_deterministic_in_eval_mode(self, untrained_model):
        untrained_model.eval()
        x = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            out1 = untrained_model(x)
            out2 = untrained_model(x)
        assert torch.allclose(out1, out2)

    def test_wrong_input_shape_raises(self, untrained_model):
        x = torch.randn(1, 1, 32, 32)  # Wrong spatial dims
        with pytest.raises(RuntimeError):
            untrained_model(x)

    def test_is_base_model(self):
        model = MNISTNet()
        assert isinstance(model, BaseModel)
        assert isinstance(model, torch.nn.Module)

    def test_has_expected_layers(self):
        model = MNISTNet()
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "fc1")
        assert hasattr(model, "fc2")

    def test_parameter_count_is_positive(self):
        model = MNISTNet()
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_name_and_description(self):
        assert MNISTNet.name == "CNN"
        assert len(MNISTNet.description) > 0
