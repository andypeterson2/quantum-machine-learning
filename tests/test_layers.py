"""Unit tests for classifiers.layers (Quadratic and Polynomial layers)."""

import torch

from classifiers.layers import Quadratic, Polynomial


class TestQuadratic:
    def test_expand_shape(self):
        x = torch.randn(4, 8)
        z = Quadratic.expand(x)
        # expanded dim = input_dim * (input_dim + 1)
        assert z.shape == (4, 8 * 9)

    def test_forward_shape(self):
        layer = Quadratic(input_dim=8, output_dim=5)
        x = torch.randn(4, 8)
        out = layer(x)
        assert out.shape == (4, 5)

    def test_gradients_flow(self):
        layer = Quadratic(input_dim=4, output_dim=3)
        x = torch.randn(2, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        for p in layer.parameters():
            assert p.grad is not None

    def test_expand_single_sample(self):
        x = torch.randn(1, 3)
        z = Quadratic.expand(x)
        assert z.shape == (1, 3 * 4)

    def test_deterministic(self):
        layer = Quadratic(input_dim=4, output_dim=2)
        layer.eval()
        x = torch.randn(3, 4)
        with torch.no_grad():
            assert torch.allclose(layer(x), layer(x))


class TestPolynomial:
    def test_forward_shape(self):
        layer = Polynomial(input_dim=8, output_dim=5)
        x = torch.randn(4, 8)
        out = layer(x)
        assert out.shape == (4, 5)

    def test_output_non_negative(self):
        """exp() output should always be positive."""
        layer = Polynomial(input_dim=4, output_dim=3)
        x = torch.randn(10, 4)
        with torch.no_grad():
            out = layer(x)
        assert (out > 0).all()

    def test_gradients_flow(self):
        layer = Polynomial(input_dim=4, output_dim=3)
        x = torch.randn(2, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_handles_negative_input(self):
        """abs(x) in forward should handle negative inputs without NaN."""
        layer = Polynomial(input_dim=4, output_dim=2)
        x = torch.tensor([[-1.0, -2.0, 0.0, 3.0]])
        with torch.no_grad():
            out = layer(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
