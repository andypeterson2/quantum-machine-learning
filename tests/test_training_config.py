"""Unit tests for TrainingConfig, HistoryEntry, and advanced training features."""


from classifiers.training_config import TrainingConfig, HistoryEntry
from classifiers.trainer import Trainer, TrainResult
from classifiers.datasets.mnist.models import MNISTNet, LinearNet
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
        assert config.regularization_fn is None
        assert config.teacher_model is None
        assert config.distill_weight == 0.5

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

    def test_train_with_regularization(self):
        loader = make_fake_train_loader(n_batches=3)

        def l2_reg(model):
            return 0.01 * sum(p.pow(2).sum() for p in model.parameters())

        config = TrainingConfig(regularization_fn=l2_reg)
        trainer = Trainer(
            model_cls=LinearNet, train_loader=loader,
            dataset="mnist", epochs=1, config=config,
        )
        result = trainer.train()
        assert isinstance(result, TrainResult)

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
