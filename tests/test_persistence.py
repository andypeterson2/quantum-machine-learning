"""Unit tests for classifiers.persistence.ModelPersistence."""

from dataclasses import replace

import pytest
import torch

from classifiers.datasets.iris.models import IrisLinear
from classifiers.datasets.mnist.models import MNISTNet
from classifiers.model_registry import ModelEntry
from classifiers.persistence import ModelPersistence


@pytest.fixture
def models_dir(tmp_path):
    return tmp_path / "models"


@pytest.fixture
def store(models_dir):
    return ModelPersistence(models_dir)


@pytest.fixture
def sample_entry():
    return ModelEntry(
        model=MNISTNet(),
        model_type="CNN",
        dataset="mnist",
        epochs=3,
        batch_size=64,
        lr=1e-3,
        training_history=[{"epoch": 0, "batch": 0, "train_loss": 0.5}],
        num_params=12345,
    )


class TestSaveAndLoad:
    def test_save_creates_file(self, store, models_dir, sample_entry):
        filename = store.save("My Model", sample_entry)
        assert (models_dir / filename).exists()
        assert filename.endswith(".pt")

    def test_save_and_load_roundtrip(self, store, sample_entry):
        filename = store.save("roundtrip", sample_entry)
        loaded = store.load(filename)
        assert loaded["name"] == "roundtrip"
        assert loaded["model_type"] == "CNN"
        assert loaded["dataset"] == "mnist"
        assert loaded["epochs"] == 3
        assert loaded["batch_size"] == 64
        assert isinstance(loaded["model"], MNISTNet)

    def test_loaded_model_produces_output(self, store, sample_entry):
        filename = store.save("test", sample_entry)
        loaded = store.load(filename)
        model = loaded["model"]
        x = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10)

    def test_save_preserves_training_history(self, store, sample_entry):
        filename = store.save("hist", sample_entry)
        loaded = store.load(filename)
        assert len(loaded["training_history"]) == 1
        assert loaded["training_history"][0]["train_loss"] == 0.5

    def test_save_preserves_num_params(self, store, sample_entry):
        filename = store.save("params", sample_entry)
        loaded = store.load(filename)
        assert loaded["num_params"] == 12345


class TestListFiles:
    def test_list_empty_dir(self, store):
        assert store.list_files() == []

    def test_list_after_save(self, store, sample_entry):
        store.save("model_a", sample_entry)
        store.save("model_b", sample_entry)
        files = store.list_files()
        assert len(files) == 2
        names = {f["name"] for f in files}
        assert "model_a" in names
        assert "model_b" in names

    def test_list_file_metadata(self, store, sample_entry):
        store.save("check", sample_entry)
        files = store.list_files()
        f = files[0]
        assert f["model_type"] == "CNN"
        assert f["dataset"] == "mnist"
        assert f["epochs"] == 3


class TestOneDatasetDoesNotOverwriteAnother:
    """Two datasets may hold a model of the same name.

    Default names are per-dataset counters, so "Model 1" exists in every
    dataset at once; a filename built from the name alone made the second
    export destroy the first.
    """

    @staticmethod
    def _iris_twin(entry):
        """The same display name on the other dataset, with an architecture it has."""
        return replace(entry, dataset="iris", model_type="Linear", model=IrisLinear())

    def test_same_name_two_datasets_are_two_files(self, store, models_dir, sample_entry):
        mnist_file = store.save("Model 1", sample_entry)
        iris_file = store.save("Model 1", self._iris_twin(sample_entry))

        assert mnist_file != iris_file
        assert (models_dir / mnist_file).exists()
        assert (models_dir / iris_file).exists()
        assert store.load(mnist_file)["dataset"] == "mnist"
        assert store.load(iris_file)["dataset"] == "iris"

    def test_both_survive_in_the_listing(self, store, sample_entry):
        store.save("Model 1", sample_entry)
        store.save("Model 1", self._iris_twin(sample_entry))
        listed = {f["dataset"] for f in store.list_files()}
        assert listed == {"mnist", "iris"}


class TestFilenameValidation:
    def test_safe_filename_replaces_spaces(self):
        assert ModelPersistence._safe_filename("My Model", "mnist") == "mnist__My_Model.pt"

    def test_safe_filename_replaces_special_chars(self):
        result = ModelPersistence._safe_filename("model/../../etc", "mnist")
        assert "/" not in result
        assert ".." not in result

    def test_safe_filename_separates_the_datasets(self):
        """The same model name under two datasets is two files."""
        mnist = ModelPersistence._safe_filename("Model 1", "mnist")
        iris = ModelPersistence._safe_filename("Model 1", "iris")
        assert mnist != iris

    def test_validate_rejects_path_traversal(self, store):
        with pytest.raises(ValueError):
            store.load("../../../etc/passwd")

    def test_validate_rejects_non_pt(self, store):
        with pytest.raises(ValueError):
            store.load("model.txt")

    def test_load_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("does_not_exist.pt")
