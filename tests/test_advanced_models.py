"""Unit tests for Quadratic and Polynomial MNIST model architectures."""

import torch

from classifiers.base_model import BaseModel
from classifiers.datasets.mnist.models import MNISTQuadraticNet, MNISTPolynomialNet


class TestQuadraticNet:
    def test_is_base_model(self):
        model = MNISTQuadraticNet()
        assert isinstance(model, BaseModel)

    def test_output_shape(self):
        model = MNISTQuadraticNet()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_name_and_description(self):
        assert MNISTQuadraticNet.name == "Quadratic"
        assert len(MNISTQuadraticNet.description) > 0

    def test_has_quad_layer(self):
        model = MNISTQuadraticNet()
        assert hasattr(model, "quad")

    def test_gradients_flow(self):
        model = MNISTQuadraticNet()
        x = torch.randn(2, 1, 28, 28)
        target = torch.randint(0, 10, (2,))
        loss = model.loss_fn(model(x), target)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestPolynomialNet:
    def test_is_base_model(self):
        model = MNISTPolynomialNet()
        assert isinstance(model, BaseModel)

    def test_output_shape(self):
        model = MNISTPolynomialNet()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_name_and_description(self):
        assert MNISTPolynomialNet.name == "Polynomial"
        assert len(MNISTPolynomialNet.description) > 0

    def test_has_polynomial_layers(self):
        model = MNISTPolynomialNet()
        assert hasattr(model, "poly1")
        assert hasattr(model, "poly2")

    def test_deterministic_in_eval(self):
        model = MNISTPolynomialNet()
        model.eval()
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))
