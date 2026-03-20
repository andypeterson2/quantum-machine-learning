"""Unit tests for classifiers.persistence.ModelPersistence."""

import pytest
import torch

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


class TestFilenameValidation:
    def test_safe_filename_replaces_spaces(self):
        assert ModelPersistence._safe_filename("My Model") == "My_Model.pt"

    def test_safe_filename_replaces_special_chars(self):
        result = ModelPersistence._safe_filename("model/../../etc")
        assert "/" not in result
        assert ".." not in result

    def test_validate_rejects_path_traversal(self, store):
        with pytest.raises(ValueError):
            store.load("../../../etc/passwd")

    def test_validate_rejects_non_pt(self, store):
        with pytest.raises(ValueError):
            store.load("model.txt")

    def test_load_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("does_not_exist.pt")
