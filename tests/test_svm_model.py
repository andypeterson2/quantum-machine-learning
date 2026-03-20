"""Unit tests for SVMNet architecture and hinge loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from classifiers.datasets.mnist.models import SVMNet, MNISTNet, LinearNet
from classifiers.base_model import BaseModel
from classifiers.losses import multi_class_hinge_loss


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def svm():
    return SVMNet()


# ── Architecture ──────────────────────────────────────────────────────────────


class TestSVMNetArchitecture:
    def test_is_base_model(self, svm):
        assert isinstance(svm, BaseModel)
        assert isinstance(svm, nn.Module)

    def test_output_shape_single(self, svm):
        out = svm(torch.randn(1, 1, 28, 28))
        assert out.shape == (1, 10)

    def test_output_shape_batch(self, svm):
        out = svm(torch.randn(32, 1, 28, 28))
        assert out.shape == (32, 10)

    def test_has_fc_layer(self, svm):
        assert hasattr(svm, "fc")
        assert isinstance(svm.fc, nn.Linear)
        assert svm.fc.in_features == 784
        assert svm.fc.out_features == 10

    def test_deterministic_in_eval_mode(self, svm):
        svm.eval()
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            assert torch.allclose(svm(x), svm(x))

    def test_name_and_description(self):
        assert SVMNet.name == "SVM"
        assert len(SVMNet.description) > 0

    def test_same_param_count_as_linear(self):
        svm_params = sum(p.numel() for p in SVMNet().parameters())
        linear_params = sum(p.numel() for p in LinearNet().parameters())
        assert svm_params == linear_params  # both are 784→10 linear layers

    def test_fewer_params_than_cnn(self):
        svm_params = sum(p.numel() for p in SVMNet().parameters())
        cnn_params = sum(p.numel() for p in MNISTNet().parameters())
        assert svm_params < cnn_params

    def test_output_is_raw_scores_not_probabilities(self, svm):
        """Forward output should NOT sum to 1 (no softmax in forward)."""
        svm.eval()
        with torch.no_grad():
            out = svm(torch.randn(4, 1, 28, 28))
        row_sums = out.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones(4), atol=1e-3)

    def test_gradients_flow(self, svm):
        x = torch.randn(4, 1, 28, 28)
        target = torch.randint(0, 10, (4,))
        loss = SVMNet.loss_fn(svm(x), target)
        loss.backward()
        for p in svm.parameters():
            assert p.grad is not None


# ── Hinge loss function ───────────────────────────────────────────────────────


class TestMultiClassHingeLoss:
    def test_returns_scalar_tensor(self):
        out = torch.randn(8, 10)
        tgt = torch.randint(0, 10, (8,))
        loss = multi_class_hinge_loss(out, tgt)
        assert loss.ndim == 0  # scalar

    def test_non_negative(self):
        out = torch.randn(16, 10)
        tgt = torch.randint(0, 10, (16,))
        assert multi_class_hinge_loss(out, tgt).item() >= 0.0

    def test_zero_when_correct_class_dominates_by_margin(self):
        """If correct class score >> all others by > margin, loss == 0."""
        out = torch.zeros(3, 10)
        tgt = torch.tensor([2, 5, 9])
        for i, c in enumerate(tgt):
            out[i, c] = 10.0
        loss = multi_class_hinge_loss(out, tgt, margin=1.0)
        assert loss.item() < 1e-5

    def test_positive_when_classes_tied(self):
        """All-zeros output means no class dominates — should have positive loss."""
        out = torch.zeros(4, 10)
        tgt = torch.zeros(4, dtype=torch.long)
        loss = multi_class_hinge_loss(out, tgt, margin=1.0)
        assert abs(loss.item() - 9.0) < 1e-4

    def test_larger_margin_increases_loss(self):
        out = torch.zeros(4, 10)
        tgt = torch.zeros(4, dtype=torch.long)
        loss1 = multi_class_hinge_loss(out, tgt, margin=1.0)
        loss2 = multi_class_hinge_loss(out, tgt, margin=2.0)
        assert loss2.item() > loss1.item()

    def test_differs_from_cross_entropy(self):
        """Hinge loss should produce different values than cross-entropy."""
        torch.manual_seed(0)
        out = torch.randn(8, 10)
        tgt = torch.randint(0, 10, (8,))
        hinge = multi_class_hinge_loss(out, tgt).item()
        ce = F.cross_entropy(out, tgt).item()
        assert abs(hinge - ce) > 1e-3

    def test_autograd_safe(self):
        """Loss backward must not raise (no in-place ops on computed tensors)."""
        out = torch.randn(4, 10, requires_grad=True)
        tgt = torch.randint(0, 10, (4,))
        loss = multi_class_hinge_loss(out, tgt)
        loss.backward()
        assert out.grad is not None


# ── loss_fn class method ─────────────────────────────────────────────────────


class TestSVMLossFn:
    def test_loss_fn_is_defined(self):
        assert hasattr(SVMNet, "loss_fn")
        assert callable(SVMNet.loss_fn)

    def test_loss_fn_overrides_base(self):
        """SVMNet.loss_fn must produce different results than BaseModel.loss_fn."""
        torch.manual_seed(42)
        out = torch.randn(8, 10)
        tgt = torch.randint(0, 10, (8,))
        svm_loss = SVMNet.loss_fn(out, tgt).item()
        base_loss = BaseModel.loss_fn(out, tgt).item()
        assert abs(svm_loss - base_loss) > 1e-3

    def test_loss_fn_matches_hinge_loss(self):
        torch.manual_seed(7)
        out = torch.randn(8, 10)
        tgt = torch.randint(0, 10, (8,))
        assert torch.allclose(
            SVMNet.loss_fn(out, tgt),
            multi_class_hinge_loss(out, tgt),
        )

    def test_base_model_loss_fn_is_cross_entropy(self):
        torch.manual_seed(0)
        out = torch.randn(8, 10)
        tgt = torch.randint(0, 10, (8,))
        assert torch.allclose(BaseModel.loss_fn(out, tgt), F.cross_entropy(out, tgt))

    def test_cnn_inherits_cross_entropy_loss(self):
        """CNN and Linear should still inherit the default CE loss from BaseModel."""
        torch.manual_seed(1)
        out = torch.randn(8, 10)
        tgt = torch.randint(0, 10, (8,))
        ce = F.cross_entropy(out, tgt)
        assert torch.allclose(MNISTNet.loss_fn(out, tgt), ce)
        assert torch.allclose(LinearNet.loss_fn(out, tgt), ce)
