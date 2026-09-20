"""Unit tests for TrainingConfig, HistoryEntry, and advanced training features."""


import pytest
import torch

from classifiers.datasets.mnist.models import LinearNet, MNISTNet
from classifiers.trainer import Trainer, TrainResult
from classifiers.training_config import HistoryEntry, TrainingConfig
from tests.conftest import make_fake_train_loader


class TestHistoryEntry:
    def test_to_dict_basic(self):
        entry = HistoryEntry(epoch=1, batch=50, train_loss=0.4)
        d = entry.to_dict()
        assert d["epoch"] == 1
        assert d["batch"] == 50
        assert d["train_loss"] == 0.4
        assert "val_accuracy" not in d

    def test_to_dict_with_val_accuracy(self):
        entry = HistoryEntry(epoch=2, batch=100, train_loss=0.3, val_accuracy=0.92)
        d = entry.to_dict()
        assert d["val_accuracy"] == 0.92

    def test_defaults(self):
        entry = HistoryEntry(epoch=0, batch=0, train_loss=1.0)
        assert entry.val_accuracy is None


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.patience is None
        assert config.val_gap == 50
        assert config.teacher_model is None
        assert config.distill_weight == 0.5
        assert config.distill_temperature == 4.0

    def test_custom_values(self):
        config = TrainingConfig(patience=5, val_gap=25, distill_weight=0.3)
        assert config.patience == 5
        assert config.val_gap == 25
        assert config.distill_weight == 0.3


class TestTrainerWithConfig:
    def test_train_with_validation(self):
        train_loader = make_fake_train_loader(n_batches=5)
        val_loader = make_fake_train_loader(batch_size=16, n_batches=2)
        config = TrainingConfig(patience=3, val_gap=2)

        trainer = Trainer(
            model_cls=MNISTNet, train_loader=train_loader,
            dataset="mnist", epochs=2, config=config,
            val_loader=val_loader,
        )
        result = trainer.train()
        assert isinstance(result, TrainResult)
        assert len(result.history) > 0

    def test_train_with_distillation(self):
        loader = make_fake_train_loader(n_batches=3)
        teacher = MNISTNet()
        teacher.eval()

        config = TrainingConfig(teacher_model=teacher, distill_weight=0.3)
        trainer = Trainer(
            model_cls=LinearNet, train_loader=loader,
            dataset="mnist", epochs=1, config=config,
        )
        result = trainer.train()
        assert isinstance(result, TrainResult)
        assert result.model_type == "Linear"


class TestDistillationLoss:
    """The distillation term is KL(teacher || student) on softened outputs, times T²."""

    def test_zero_when_student_matches_teacher(self):
        from classifiers.trainer import distillation_loss

        logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.3, -2.0]])
        assert distillation_loss(logits, logits.clone(), 4.0).item() == pytest.approx(0.0, abs=1e-6)

    def test_matches_the_formula(self):
        from classifiers.trainer import distillation_loss

        s = torch.tensor([[1.0, 0.0, -1.0]])
        t = torch.tensor([[0.0, 2.0, 0.0]])
        temp = 2.0
        p = torch.softmax(t / temp, dim=1)
        q = torch.log_softmax(s / temp, dim=1)
        expected = (p * (p.log() - q)).sum() * temp * temp
        assert distillation_loss(s, t, temp).item() == pytest.approx(expected.item(), rel=1e-5)

    def test_only_the_ranking_matters_not_the_logit_offset(self):
        from classifiers.trainer import distillation_loss

        s = torch.tensor([[1.0, 0.0, -1.0]])
        t = torch.tensor([[0.0, 2.0, 0.0]])
        shifted = distillation_loss(s + 5.0, t - 3.0, 4.0).item()
        assert shifted == pytest.approx(distillation_loss(s, t, 4.0).item(), rel=1e-5)


class TestEarlyStoppingCountsValidationChecks:
    """Patience is measured in the unit it is counted in.

    Validation runs every ``val_gap`` batches, so an epoch holds many checks.
    Counting patience in epochs meant the rule could not fire until a whole
    epoch of checks had passed, and then only one epoch later still.
    """

    @staticmethod
    def _trainer(monkeypatch, accuracies, *, patience, val_gap=1, epochs=3):
        """A trainer whose validation returns *accuracies* in order."""
        scripted = iter(accuracies)
        monkeypatch.setattr(
            Trainer, "_validate", lambda self, model: next(scripted, accuracies[-1])
        )
        return Trainer(
            model_cls=LinearNet,
            train_loader=make_fake_train_loader(batch_size=8, n_batches=5),
            dataset="mnist",
            epochs=epochs,
            config=TrainingConfig(patience=patience, val_gap=val_gap),
            val_loader=make_fake_train_loader(batch_size=8, n_batches=1),
        )

    def test_stops_inside_the_first_epoch(self, monkeypatch):
        """One good check then two flat ones, with five batches in the epoch."""
        trainer = self._trainer(monkeypatch, [0.9, 0.9, 0.9, 0.9, 0.9], patience=2)
        result = trainer.train()
        assert result.stopped_early
        assert result.epochs_completed == 1

    def test_improvement_resets_the_count(self, monkeypatch):
        trainer = self._trainer(
            monkeypatch, [0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9], patience=2, epochs=2
        )
        result = trainer.train()
        assert result.best_val_accuracy == pytest.approx(0.9)

    def test_a_model_below_the_floor_is_never_stopped(self, monkeypatch):
        """The floor is what keeps a run still near chance training."""
        trainer = self._trainer(monkeypatch, [0.5] * 20, patience=1)
        result = trainer.train()
        assert not result.stopped_early
        assert result.epochs_completed == 3

    def test_the_message_names_the_unit(self, monkeypatch):
        trainer = self._trainer(monkeypatch, [0.9] * 10, patience=2)
        messages: list[str] = []
        trainer.train(on_status=lambda m: messages.append(m) if isinstance(m, str) else None)
        assert any("validation checks" in m for m in messages)
