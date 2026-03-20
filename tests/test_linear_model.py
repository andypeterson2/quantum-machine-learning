"""Unit tests for LinearNet architecture."""

import torch

from classifiers.datasets.mnist.models import LinearNet, MNISTNet
from classifiers.base_model import BaseModel


class TestLinearNet:
    def test_output_shape_single(self, untrained_linear):
        x = torch.randn(1, 1, 28, 28)
        out = untrained_linear(x)
        assert out.shape == (1, 10)

    def test_output_shape_batch(self, untrained_linear):
        x = torch.randn(16, 1, 28, 28)
        out = untrained_linear(x)
        assert out.shape == (16, 10)

    def test_is_base_model(self):
        model = LinearNet()
        assert isinstance(model, BaseModel)
        assert isinstance(model, torch.nn.Module)

    def test_has_fc_layer(self):
        model = LinearNet()
        assert hasattr(model, "fc")
        assert isinstance(model.fc, torch.nn.Linear)
        assert model.fc.in_features == 784
        assert model.fc.out_features == 10

    def test_deterministic_in_eval_mode(self, untrained_linear):
        untrained_linear.eval()
        x = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            out1 = untrained_linear(x)
            out2 = untrained_linear(x)
        assert torch.allclose(out1, out2)

    def test_name_and_description(self):
        assert LinearNet.name == "Linear"
        assert len(LinearNet.description) > 0

    def test_fewer_params_than_cnn(self):
        """Linear model should have far fewer parameters than CNN."""
        linear_params = sum(p.numel() for p in LinearNet().parameters())
        cnn_params = sum(p.numel() for p in MNISTNet().parameters())
        assert linear_params < cnn_params
