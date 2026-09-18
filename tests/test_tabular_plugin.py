"""The pipeline Iris and BB84 share, tested once.

Both plugins used to carry their own copy of it — standardise on the training
statistics, cache, split 80/20, and apply the same transformation at inference.
The copies agreed, so the tests only ever checked each dataset's version of the
same behaviour. This tests the behaviour itself, on a plugin small enough to
compute by hand.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest
import torch

from classifiers.base_model import BaseModel
from classifiers.datasets.tabular import TabularPlugin


class _ToyPlugin(TabularPlugin):
    """Ten training rows and two test rows, with known statistics."""

    name = "toy"
    display_name = "Toy"
    num_classes = 2
    class_labels: typing.ClassVar[list[str]] = ["low", "high"]
    feature_names: typing.ClassVar[list[str]] = ["a", "b"]

    def load_raw(self):
        train_x = np.array([[float(i), float(2 * i)] for i in range(10)], dtype=np.float32)
        train_y = np.array([i % 2 for i in range(10)], dtype=np.int64)
        # Far outside the training range: if these leaked into the statistics,
        # the standardised training rows would shift.
        test_x = np.array([[100.0, 200.0], [-100.0, -200.0]], dtype=np.float32)
        test_y = np.array([1, 0], dtype=np.int64)
        return train_x, train_y, test_x, test_y

    def get_model_types(self) -> dict[str, type[BaseModel]]:
        return {}


@pytest.fixture
def plugin() -> _ToyPlugin:
    return _ToyPlugin()


def _rows(loader) -> int:
    return sum(len(y) for _, y in loader)


class TestSplits:
    def test_training_rows_split_eighty_twenty(self, plugin) -> None:
        assert _rows(plugin.get_train_loader(4)) == 8
        assert _rows(plugin.get_val_loader(4)) == 2

    def test_validation_is_the_tail_of_the_training_rows(self, plugin) -> None:
        """Not a random slice: the exporter's feature ranges and the trainer's
        early stopping both rely on this being stable."""
        val_x = torch.cat([x for x, _ in plugin.get_val_loader(4)])
        expected = (torch.tensor([[8.0, 16.0], [9.0, 18.0]]) - plugin._mean) / plugin._std
        assert torch.allclose(val_x, expected)

    def test_the_test_split_is_served_whole(self, plugin) -> None:
        assert _rows(plugin.get_test_loader(4)) == 2

    def test_train_and_validation_do_not_overlap(self, plugin) -> None:
        train_x = torch.cat([x for x, _ in plugin.get_train_loader(4)])
        val_x = torch.cat([x for x, _ in plugin.get_val_loader(4)])
        for row in val_x:
            assert not any(torch.allclose(row, other) for other in train_x)


class TestStandardisation:
    def test_statistics_come_from_the_training_rows_only(self, plugin) -> None:
        """The test rows here are two orders of magnitude out; if they informed
        the mean, it would not be 4.5."""
        mean, std = plugin.normalization()
        assert mean == pytest.approx([4.5, 9.0])
        assert std == pytest.approx(np.std(np.arange(10), ddof=1) * np.array([1.0, 2.0]), abs=1e-5)

    def test_inference_applies_the_same_transformation(self, plugin) -> None:
        """A sample through preprocess must land where the same row lands in
        the loaders — the reason the constants are published with the weights."""
        mean, std = plugin.normalization()
        got = plugin.preprocess({"a": 3.0, "b": 6.0})
        expected = (torch.tensor([[3.0, 6.0]]) - torch.tensor(mean)) / torch.tensor(std)
        assert torch.allclose(got, expected)

    def test_a_constant_feature_does_not_divide_by_zero(self) -> None:
        class _Constant(_ToyPlugin):
            def load_raw(self):
                train_x = np.ones((6, 2), dtype=np.float32)
                train_y = np.array([0, 1] * 3, dtype=np.int64)
                return train_x, train_y, train_x[:2], train_y[:2]

        standardised = torch.cat([x for x, _ in _Constant().get_train_loader(2)])
        assert torch.isfinite(standardised).all()

    def test_the_dataset_is_loaded_once(self, plugin, monkeypatch) -> None:
        calls = {"n": 0}
        original = plugin.load_raw

        def counted():
            calls["n"] += 1
            return original()

        monkeypatch.setattr(plugin, "load_raw", counted)
        plugin.get_train_loader(4)
        plugin.get_test_loader(4)
        plugin.normalization()
        assert calls["n"] == 1
