"""Unit tests for classifiers.model_registry.ModelRegistry."""


from classifiers.datasets.mnist.models import MNISTNet, LinearNet
from classifiers.model_registry import ModelRegistry, ModelEntry

DS = "mnist"  # Dataset slug used throughout tests


class TestModelEntry:
    def test_defaults(self):
        model = MNISTNet()
        entry = ModelEntry(
            model=model, model_type="CNN", dataset=DS,
            epochs=3, batch_size=64, lr=1e-3,
        )
        assert entry.eval_result is None
        assert entry.epochs == 3
        assert entry.model_type == "CNN"
        assert entry.dataset == DS

    def test_linear_entry(self):
        model = LinearNet()
        entry = ModelEntry(
            model=model, model_type="Linear", dataset=DS,
            epochs=5, batch_size=32, lr=0.01,
        )
        assert entry.model_type == "Linear"


class TestModelRegistry:
    def test_starts_empty(self):
        reg = ModelRegistry()
        assert len(reg) == 0
        assert reg.names(DS) == []
        assert reg.items(DS) == []

    def test_add_and_get(self, untrained_model):
        reg = ModelRegistry()
        reg.add(DS, "m1", untrained_model, model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        assert len(reg) == 1
        entry = reg.get(DS, "m1")
        assert entry is not None
        assert entry.model is untrained_model
        assert entry.model_type == "CNN"
        assert entry.lr == 0.01

    def test_get_missing_returns_none(self):
        reg = ModelRegistry()
        assert reg.get(DS, "nonexistent") is None

    def test_remove(self, untrained_model):
        reg = ModelRegistry()
        reg.add(DS, "m1", untrained_model, model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        reg.remove(DS, "m1")
        assert len(reg) == 0
        assert reg.get(DS, "m1") is None

    def test_remove_missing_is_noop(self):
        reg = ModelRegistry()
        reg.remove(DS, "nonexistent")  # Should not raise
        assert len(reg) == 0

    def test_names(self, untrained_model):
        reg = ModelRegistry()
        reg.add(DS, "alpha", untrained_model, model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        reg.add(DS, "beta", untrained_model, model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        assert set(reg.names(DS)) == {"alpha", "beta"}

    def test_items(self, untrained_model):
        reg = ModelRegistry()
        reg.add(DS, "m1", untrained_model, model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        items = reg.items(DS)
        assert len(items) == 1
        name, entry = items[0]
        assert name == "m1"
        assert isinstance(entry, ModelEntry)

    def test_next_name_auto_increments(self):
        reg = ModelRegistry()
        assert reg.next_name(DS) == "Model 1"
        assert reg.next_name(DS) == "Model 2"
        assert reg.next_name(DS) == "Model 3"

    def test_add_overwrites_existing(self, untrained_model):
        reg = ModelRegistry()
        model2 = MNISTNet()
        reg.add(DS, "m1", untrained_model, model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        reg.add(DS, "m1", model2, model_type="CNN", epochs=2, batch_size=64, lr=0.001)
        assert len(reg) == 1
        assert reg.get(DS, "m1").model is model2
        assert reg.get(DS, "m1").epochs == 2

    def test_multiple_models(self):
        reg = ModelRegistry()
        models = [MNISTNet() for _ in range(5)]
        for i, m in enumerate(models):
            reg.add(DS, f"model_{i}", m, model_type="CNN", epochs=i + 1, batch_size=32, lr=0.01)
        assert len(reg) == 5
        assert reg.get(DS, "model_3").epochs == 4

    def test_mixed_model_types(self):
        reg = ModelRegistry()
        reg.add(DS, "cnn", MNISTNet(), model_type="CNN", epochs=3, batch_size=64, lr=0.001)
        reg.add(DS, "linear", LinearNet(), model_type="Linear", epochs=5, batch_size=128, lr=0.01)
        assert len(reg) == 2
        assert reg.get(DS, "cnn").model_type == "CNN"
        assert reg.get(DS, "linear").model_type == "Linear"

    def test_dataset_isolation(self):
        """Models in different datasets don't collide."""
        reg = ModelRegistry()
        reg.add("mnist", "m1", MNISTNet(), model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        reg.add("iris", "m1", LinearNet(), model_type="Linear", epochs=1, batch_size=32, lr=0.01)
        assert len(reg) == 2
        assert reg.get("mnist", "m1").model_type == "CNN"
        assert reg.get("iris", "m1").model_type == "Linear"

    def test_update_eval_result(self, untrained_model):
        from classifiers.evaluator import EvalResult

        reg = ModelRegistry()
        reg.add(DS, "m1", untrained_model, model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        ev = EvalResult(accuracy=0.95, avg_loss=0.1)
        reg.update_eval_result(DS, "m1", ev)
        assert reg.get(DS, "m1").eval_result is ev
