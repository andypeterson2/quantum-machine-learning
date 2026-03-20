"""Unit tests for classifiers.trainer.Trainer and TrainResult."""


from classifiers.trainer import Trainer, TrainResult
from classifiers.datasets.mnist.models import MNISTNet, LinearNet
from tests.conftest import make_fake_train_loader


class TestTrainResult:
    def test_fields(self):
        model = MNISTNet()
        result = TrainResult(
            model=model, model_type="CNN", dataset="mnist",
            epochs=3, batch_size=64, lr=1e-3,
        )
        assert result.model is model
        assert result.model_type == "CNN"
        assert result.dataset == "mnist"
        assert result.epochs == 3
        assert result.batch_size == 64
        assert result.lr == 1e-3


class TestTrainer:
    def test_init_stores_params(self):
        loader = make_fake_train_loader()
        trainer = Trainer(
            model_cls=MNISTNet, train_loader=loader,
            dataset="mnist", epochs=5, lr=0.01,
        )
        assert trainer.model_cls is MNISTNet
        assert trainer.epochs == 5
        assert trainer.lr == 0.01
        assert trainer.dataset == "mnist"

    def test_train_returns_train_result(self):
        loader = make_fake_train_loader(n_batches=3)
        trainer = Trainer(
            model_cls=MNISTNet, train_loader=loader,
            dataset="mnist", epochs=1,
        )
        result = trainer.train()
        assert isinstance(result, TrainResult)
        assert isinstance(result.model, MNISTNet)
        assert result.model_type == "CNN"
        assert result.epochs == 1

    def test_train_with_linear_model(self):
        loader = make_fake_train_loader(n_batches=3)
        trainer = Trainer(
            model_cls=LinearNet, train_loader=loader,
            dataset="mnist", epochs=1,
        )
        result = trainer.train()
        assert isinstance(result, TrainResult)
        assert isinstance(result.model, LinearNet)
        assert result.model_type == "Linear"

    def test_train_calls_status_callback(self):
        loader = make_fake_train_loader(n_batches=3)
        statuses = []
        trainer = Trainer(
            model_cls=MNISTNet, train_loader=loader,
            dataset="mnist", epochs=1,
        )
        trainer.train(on_status=statuses.append)
        assert len(statuses) > 0
        # Should have a "Preparing" message and a "complete" message
        str_statuses = [s for s in statuses if isinstance(s, str)]
        assert any("Preparing" in s for s in str_statuses)
        assert any("complete" in s.lower() for s in str_statuses)

    def test_train_result_hyperparams_match(self):
        loader = make_fake_train_loader(n_batches=3)
        trainer = Trainer(
            model_cls=MNISTNet, train_loader=loader,
            dataset="mnist", epochs=2, lr=0.005,
        )
        result = trainer.train()
        assert result.epochs == 2
        assert result.lr == 0.005

    def test_train_no_status_callback(self):
        loader = make_fake_train_loader(n_batches=3)
        trainer = Trainer(
            model_cls=MNISTNet, train_loader=loader,
            dataset="mnist", epochs=1,
        )
        result = trainer.train(on_status=None)
        assert isinstance(result, TrainResult)

    def test_train_result_has_num_params(self):
        loader = make_fake_train_loader(n_batches=3)
        trainer = Trainer(
            model_cls=MNISTNet, train_loader=loader,
            dataset="mnist", epochs=1,
        )
        result = trainer.train()
        assert result.num_params > 0

    def test_train_result_epochs_completed(self):
        loader = make_fake_train_loader(n_batches=3)
        trainer = Trainer(
            model_cls=MNISTNet, train_loader=loader,
            dataset="mnist", epochs=2,
        )
        result = trainer.train()
        assert result.epochs_completed == 2
